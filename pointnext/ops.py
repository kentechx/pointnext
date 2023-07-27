# This is adapted from https://github.com/guochengqian/openpoints/blob/2bc0bf9cb2aee0fcd61f6cdc3abca1207e5e809e/models/layers/subsample.py
from typing import Tuple

import torch
from torch.autograd import Function
from pathlib import Path
torch.ops.load_library(Path(__file__).parent / '_C.so')
_C = torch.ops.my_ops
# from pointnext import _C


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (b, n, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, m) tensor containing the set (idx)
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.zeros(B, npoint, dtype=torch.int32, device=xyz.device)
        temp = torch.full((B, N), fill_value=1e10, dtype=torch.float32, device=xyz.device)

        _C.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return FurthestPointSampling.forward(torch.zeros(1), xyz, npoint)
    else:
        return FurthestPointSampling.apply(xyz, npoint)


class BallQuery(Function):
    @staticmethod
    def forward(ctx, src: torch.Tensor, query: torch.Tensor, radius: float, k: int) -> torch.Tensor:
        """
        :param ctx:
        :param src: (b, n, 3) xyz coordinates of the features
        :param query: (b, m, 3) centers of the ball query
        :param radius: float, radius of the balls
        :param k: int, maximum number of features in the balls
        :return:
            idx: (b, m, k) tensor with the indicies of the features that form the query balls
        """
        assert src.is_contiguous()
        assert query.is_contiguous()

        b, n, _ = src.size()
        m = query.size(1)
        # idx = torch.cuda.IntTensor(b, m, k, device=src.device).zero_()
        idx = torch.zeros(b, m, k, dtype=torch.int32, device=src.device)
        _C.ball_query_wrapper(b, n, m, radius, k, query, src, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


def ball_query(src: torch.Tensor, query: torch.Tensor, radius: float, k: int) -> torch.Tensor:
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return BallQuery.forward(torch.zeros(1), src, query, radius, k)
    else:
        return BallQuery.apply(src, query, radius, k)


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the three nearest neighbors of unknown in known
        :param ctx:
        :param unknown: (B, N, 3)
        :param known: (B, M, 3)
        :return:
            dist: (B, N, 3) l2 distance to the three nearest neighbors
            idx: (B, N, 3) index of 3 nearest neighbors
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()

        B, N, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        _C.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


def three_nn(unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return ThreeNN.forward(torch.zeros(1), unknown, known)
    else:
        return ThreeNN.apply(unknown, known)


class ThreeInterpolate(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Performs weight linear interpolation on 3 features
        :param ctx:
        :param features: (B, C, M) Features descriptors to be interpolated from
        :param idx: (B, n, 3) three nearest neighbors of the target features in features
        :param weight: (B, n, 3) weights
        :return:
            output: (B, C, N) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()

        B, c, m = features.size()
        n = idx.size(1)
        ctx.three_interpolate_for_backward = (idx, weight, m)
        output = torch.cuda.FloatTensor(B, c, n)

        _C.three_interpolate_wrapper(B, c, m, n, features, idx, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, N) tensor with gradients of outputs
        :return:
            grad_features: (B, C, M) tensor with gradients of features
            None:
            None:
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        grad_features = torch.zeros([B, c, m], device='cuda', requires_grad=True)
        grad_out_data = grad_out.data.contiguous()

        _C.three_interpolate_grad_wrapper(B, c, n, m, grad_out_data, idx, weight, grad_features.data)
        return grad_features, None, None


def three_interpolate(features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return ThreeInterpolate.forward(torch.zeros(1), features, idx, weight)
    else:
        return ThreeInterpolate.apply(features, idx, weight)


def three_interpolation(known_xyz, know_feat, unknown_xyz):
    """
    :param known_xyz: (b, m, 3)
    :param know_feat: (b, c, m)
    :param unknown_xyz: (b, n, 3)
    output: (b, n, c)
    """
    dist, idx = three_nn(unknown_xyz, known_xyz)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    interpolated_feats = three_interpolate(know_feat, idx, weight)
    return interpolated_feats