# This is adapted from https://github.com/guochengqian/openpoints/blob/2bc0bf9cb2aee0fcd61f6cdc3abca1207e5e809e/models/layers/subsample.py
import torch
from torch.autograd import Function
from pointnext import _C


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
        # output = torch.cuda.IntTensor(B, npoint, device=xyz.device)
        # temp = torch.cuda.FloatTensor(B, N, device=xyz.device).fill_(1e10)
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        _C.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


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
        idx = torch.cuda.IntTensor(b, m, k, device=src.device).zero_()
        _C.ball_query_wrapper(b, n, m, radius, k, query, src, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply
