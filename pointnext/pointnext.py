from collections import namedtuple
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange
# from .utils import farthest_point_sampling, ball_query_pytorch
from .ops import ball_query, furthest_point_sample, three_interpolation



def exists(val):
    return val is not None


def default(*vals):
    for val in vals:
        if exists(val):
            return val


SampleResult = namedtuple('SampleResult', ['x', 'xyz', 'sample_idx', 'neighbor_idx'])


def downsample_fps(xyz, n_sample):
    # xyz: (b, 3, n)
    if n_sample == xyz.shape[-1]:
        sample_idx = torch.arange(n_sample, device=xyz.device)
        sample_idx = repeat(sample_idx, 'n -> b n', b=xyz.shape[0])
        return SampleResult(None, xyz.clone(), sample_idx, None)
    _xyz = rearrange(xyz, 'b d n -> b n d').contiguous()
    sample_idx = furthest_point_sample(_xyz, n_sample).long()  # (b, k)
    sample_xyz = xyz.gather(-1, repeat(sample_idx, 'b k -> b d k', d=xyz.shape[1]))  # (b, 3, k)
    return SampleResult(None, sample_xyz, sample_idx, None)


def _ball_query(src, query, radius, k):
    # conduct ball query on dim 1
    src = rearrange(src, 'b d n -> b n d').contiguous()
    query = rearrange(query, 'b d m -> b m d').contiguous()
    idx = ball_query(src, query, radius, k).long()
    dists = None
    return idx, dists


def cdist(x, y=None):
    # perform cdist in dimension 1
    # x: (b, d, n)
    # y: (b, d, m)
    if exists(y):
        x = rearrange(x, 'b d n -> b n d')
        y = rearrange(y, 'b d m -> b m d')
        return torch.cdist(x, y)
    else:
        x = rearrange(x, 'b d n -> b n d')
        return torch.cdist(x, x)


def gather(x, idx):
    # x: (b, d, n)
    # idx: (b, m, k)
    # output: (b, d, m, k)
    m = idx.shape[1]
    ind = repeat(idx, 'b m k -> b d (m k)', d=x.shape[1])
    out = x.gather(-1, ind)  # (b, d, (m k))
    out = rearrange(out, 'b d (m k) -> b d m k', m=m)
    return out


class SABlock(nn.Module):
    """
    Set abstraction block without downsampling.
    """

    def __init__(self, in_dim, out_dim, stride=1, layers=1, radius=0.1, k=16):
        super().__init__()
        self.stride = stride
        self.radius = radius
        self.layers = layers
        self.k = k

        dims = [in_dim + 3] + [out_dim] * layers

        if layers == 1:
            self.convs = nn.Conv2d(dims[0], dims[1], 1, bias=False)
            self.norm = nn.BatchNorm1d(out_dim)
            self.act = nn.ReLU()
        else:
            self.skip_conv = nn.Conv1d(in_dim, out_dim, 1, bias=False) if in_dim != out_dim else nn.Identity()
            self.convs = nn.Sequential(*[
                nn.Sequential(nn.Conv2d(in_d, out_d, 1, bias=False),
                              nn.BatchNorm2d(out_d),
                              nn.ReLU())
                for in_d, out_d in zip(dims[:-2], dims[1:-1])
            ])
            self.convs.append(nn.Conv2d(dims[-2], dims[-1], 1, bias=False))
            self.norm = nn.BatchNorm1d(out_dim)
            self.act = nn.ReLU()

    def route(self, src_x, src_xyz, xyz, radius, k, neighbor_idx=None):
        # src_x: (b, d, n)
        # src_xyz: (b, 3, n)
        # xyz: (b, 3, m)
        if not exists(neighbor_idx):
            neighbor_idx = _ball_query(src_xyz, xyz, radius, k)[0]  # (b, m, k)
        neighbor_xyz = gather(src_xyz, neighbor_idx)  # (b, 3, m, k)
        neighbor_xyz -= xyz[..., None]
        neighbor_xyz /= radius
        x = gather(src_x, neighbor_idx)  # (b, d, m, k)
        x = torch.cat([x, neighbor_xyz], dim=1)  # (b, d+3, m, k)
        return SampleResult(x, xyz, None, neighbor_idx)

    def forward(self, x, xyz):
        # x: (b, d, n)
        # xyz: (b, 3, n)
        # out: (b, d', n // stride)
        sample = downsample_fps(xyz, n_sample=xyz.shape[-1] // self.stride)
        inputs = x.gather(-1, repeat(sample.sample_idx, 'b k -> b d k', d=x.shape[1]))
        sample = self.route(x, xyz, sample.xyz, self.radius, self.k)
        x = self.convs(sample.x)
        x = x.max(dim=-1)[0]
        if hasattr(self, 'skip_conv'):
            x = self.skip_conv(inputs) + x
        x = self.act(self.norm(x))
        return SampleResult(x, sample.xyz, sample.sample_idx, sample.neighbor_idx)


class InvResMLP(nn.Module):

    def __init__(self, in_dim, expansion=4, radius=0.1, k=16):
        super().__init__()
        self.sa_conv = SABlock(in_dim, in_dim, stride=1, layers=1, radius=radius, k=k)

        dims = [in_dim, in_dim * expansion, in_dim]
        self.conv = nn.Sequential(
            nn.Conv1d(dims[0], dims[1], 1, bias=False),
            nn.BatchNorm1d(dims[1]),
            nn.ReLU(),
            nn.Conv1d(dims[1], dims[2], 1, bias=False),
            nn.BatchNorm1d(dims[2])
        )
        self.act = nn.ReLU()

    def forward(self, x, xyz):
        inputs = x
        x = self.sa_conv(x, xyz).x
        x = self.conv(x)
        x = self.act(inputs + x)
        return x


class UpBlock(nn.Module):

    def __init__(self, in_dim, out_dim, k=3, eps=1e-5):
        super().__init__()
        self.k = k
        assert k == 3, "only support k=3"
        self.eps = eps
        dims = [in_dim, out_dim, out_dim]
        self.conv = nn.Sequential(*[
            nn.Sequential(nn.Conv1d(in_d, out_d, 1, bias=False),
                          nn.BatchNorm1d(out_d),
                          nn.ReLU())
            for in_d, out_d in zip(dims[:-1], dims[1:])
        ])

    def route(self, src_x, src_xyz, dst_x, dst_xyz, neighbor_idx=None, dists=None):
        # use knn and weighted average to get the features
        src_xyz = rearrange(src_xyz, 'b d n -> b n d').contiguous()
        dst_xyz = rearrange(dst_x, 'b d m -> b m d').contiguous()
        lerp_x = three_interpolation(src_xyz, src_x, dst_xyz)
        dst_x = torch.cat([dst_x, lerp_x], dim=1)  # (b, d+d', m)
        return dst_x

    def forward(self, x, xyz, sub_x, sub_xyz):
        x = self.route(sub_x, sub_xyz, x, xyz)
        x = self.conv(x)
        return x


class PointNextEncoder(nn.Module):

    def __init__(
            self,
            in_dim=3,
            dims=[32, 64, 128, 256, 512],  # dims[0] is the dim of the stem output
            blocks=[4, 7, 4, 4],  # blocks: sa + invres
            strides=[4, 4, 4, 4],
            radius=0.1,
            k=32,
            sa_layers=1,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, dims[0], 1, bias=False),
            nn.BatchNorm1d(dims[0]),
            nn.ReLU()
        )

        radius_scaling = 2
        radii = [radius * (radius_scaling ** i) for i in range(len(blocks))]
        self.encoder = nn.ModuleList()
        for i in range(len(blocks)):
            layers = nn.Sequential(
                SABlock(dims[i], dims[i + 1], stride=strides[i], layers=sa_layers, radius=radii[i], k=k),
                *[InvResMLP(dims[i + 1], radius=radii[i] * radius_scaling, k=k) for _ in range(blocks[i] - 1)]
            )
            self.encoder.append(layers)

        self.out_dim = dims[-1]

    def forward_features(self, x, xyz):
        x = self.stem(x)
        features = [(x, xyz)]
        for block in self.encoder:
            sample = block[0](x, xyz)
            x, xyz = sample.x, sample.xyz
            for layer in block[1:]:
                x = layer(x, xyz)
            features.append((x, xyz))
        return features

    def forward(self, x, xyz):
        return self.forward_features(x, xyz)


class PointNextDecoder(nn.Module):

    def __init__(self, encoder_dims=[32, 64, 128, 256, 512]):
        super().__init__()
        self.decoder = nn.ModuleList()

        decoder_dims = encoder_dims[::-1]
        for i in range(len(decoder_dims) - 1):
            self.decoder.append(UpBlock(decoder_dims[i] + decoder_dims[i + 1], decoder_dims[i + 1]))

        self.out_dim = decoder_dims[-1]

    def forward(self, feats):
        sub_x, sub_xyz = feats.pop()
        for i, block in enumerate(self.decoder):
            x, xyz = feats.pop()
            x = block(x, xyz, sub_x, sub_xyz)
            sub_x, sub_xyz = x, xyz
        return x


class PointNext(nn.Module):

    def __init__(self, out_dim, encoder: PointNextEncoder, decoder: PointNextDecoder = None, n_category=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        feat_dim = decoder.out_dim if exists(decoder) else encoder.out_dim

        if n_category > 0:
            self.category_emb = nn.Embedding(n_category, feat_dim)

        self.head = nn.Conv1d(feat_dim, out_dim, 1)

    def forward(self, x, xyz, category=None):
        feats = self.encoder(x, xyz)
        if exists(self.decoder):
            out = self.decoder(feats)
        else:
            out = feats[-1][0]
        if exists(category):
            out = out + self.category_emb(category)[:, :, None]
        out = self.head(out)
        return out
