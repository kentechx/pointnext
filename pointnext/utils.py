import torch
from einops import rearrange, repeat


def exists(val):
    return val is not None


def farthest_point_sampling(x: torch.Tensor, n_sample: int, start_idx: int = None):
    # x: (b, n, 3)
    b, n = x.shape[:2]
    assert n_sample <= n, "not enough points to sample"

    if n_sample == n:
        return repeat(torch.arange(n_sample, dtype=torch.long, device=x.device), 'm -> b m', b=b)

    # start index
    if exists(start_idx):
        sel_idx = torch.full((b, n_sample), start_idx, dtype=torch.long, device=x.device)
    else:
        sel_idx = torch.randint(n, (b, n_sample), dtype=torch.long, device=x.device)

    cur_x = rearrange(x[torch.arange(b), sel_idx[:, 0]], 'b c -> b 1 c')
    min_dists = torch.full((b, n), dtype=x.dtype, device=x.device, fill_value=float('inf'))
    for i in range(1, n_sample):
        # update distance
        dists = torch.linalg.norm(x - cur_x, dim=-1)
        min_dists = torch.minimum(dists, min_dists)

        # take the farthest
        idx_farthest = torch.max(min_dists, dim=-1).indices
        sel_idx[:, i] = idx_farthest
        cur_x[:, 0, :] = x[torch.arange(b), idx_farthest]

    return sel_idx


def ball_query_pytorch(src, query, radius, k):
    # src: (b, n, 3)
    # query: (b, m, 3)
    b, n = src.shape[:2]
    m = query.shape[1]
    dists = torch.cdist(query, src)  # (b, m, n)
    idx = repeat(torch.arange(n, device=src.device), 'n -> b m n', b=b, m=m)
    idx = torch.where(dists > radius, n, idx)
    idx = idx.sort(dim=-1).values[:, :, :k]  # (b, m, k)
    idx = torch.where(idx == n, idx[:, :, [0]], idx)
    _dists = dists.gather(-1, idx)  # (b, m, k)
    return idx, _dists
