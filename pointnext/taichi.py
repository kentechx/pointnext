import taichi as ti
import torch


@ti.kernel
def _ball_query_kernel(
        src: ti.types.ndarray(ndim=3),
        query: ti.types.ndarray(ndim=3),
        out: ti.types.ndarray(ndim=3),
        dists: ti.types.ndarray(ndim=3),
        radius: ti.float32,
        K: ti.int32
):
    B, M, D = query.shape
    N = src.shape[1]

    for b, m in ti.ndrange(B, M):
        query_pt = ti.math.vec3(query[b, m, 0], query[b, m, 1], query[b, m, 2])

        count = 0
        for i in range(N):
            if count >= K:
                break
            src_pt = ti.math.vec3(src[b, i, 0], src[b, i, 1], src[b, i, 2])
            dist = (query_pt - src_pt).norm()
            if dist <= radius:
                out[b, m, count] = i
                dists[b, m, count] = dist
                count += 1
                if count == K:
                    break


def ball_query(src: torch.Tensor, query: torch.Tensor, radius, k):
    assert src.shape[-1] == 3, "src shape should be (B, N, 3)"
    idx = torch.full((*query.shape[:2], k), fill_value=-1, dtype=torch.long, device='cuda')
    dists = torch.full((*query.shape[:2], k), fill_value=-1, dtype=src.dtype, device='cuda')
    _ball_query_kernel(src.contiguous(), query.contiguous(), idx, dists, radius, k)
    mask = idx >= 0
    idx = torch.where(mask, idx, idx[:, :, [0]])
    dists = torch.where(mask, dists, dists[:, :, [0]])
    return idx, dists
