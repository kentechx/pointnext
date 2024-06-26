<img src="./pointnext.jpeg" width="1200px"></img>

[![PyPI version](https://badge.fury.io/py/pointnext.svg)](https://badge.fury.io/py/pointnext)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# pointnext

Pytorch implementation of PointNext.

## Installation

```bash
pip install pointnext
```

This will compile the CUDA operators. Please make sure that the CUDA version is compatible with your Pytorch version.

## Usage

Classification

```python
import torch
import torch.nn as nn
from pointnext import PointNext, pointnext_s


class Model(nn.Module):

    def __init__(self, in_dim=6, out_dim=40, dropout=0.5):
        super().__init__()

        encoder = pointnext_s(in_dim=in_dim)
        self.backbone = PointNext(1024, encoder=encoder)

        self.norm = nn.BatchNorm1d(1024)
        self.act = nn.ReLU()
        self.cls_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, x, xyz):
        out = self.norm(self.backbone(x, xyz))
        out = out.mean(dim=-1)
        out = self.act(out)
        out = self.cls_head(out)
        return out


model = Model(in_dim=6, out_dim=40).cuda()
xyz = torch.randn(2, 3, 1024).cuda()
rgb = torch.rand(2, 3, 1024).cuda()
x = torch.cat([xyz, rgb], dim=1)
out = model(x, xyz)
print(out.shape)  # (2, 40)
```

Semantic segmentation

```python
import torch
from pointnext import PointNext, PointNextDecoder, pointnext_s

encoder = pointnext_s(in_dim=3)
model = PointNext(40, encoder=encoder, decoder=PointNextDecoder(encoder_dims=encoder.encoder_dims)).cuda()
x = torch.randn(2, 3, 1024).cuda()
xyz = torch.randn(2, 3, 1024).cuda()
out = model(x, xyz)
```

Part segmentation

```python
import torch
from pointnext import PointNext, PointNextDecoder, pointnext_s

encoder = pointnext_s(in_dim=3)
model = PointNext(40, encoder=encoder, decoder=PointNextDecoder(encoder_dims=encoder.encoder_dims),
                  n_category=16).cuda()
x = torch.randn(2, 3, 1024).cuda()
xyz = torch.randn(2, 3, 1024).cuda()
category = torch.randint(0, 16, (2,)).cuda()
out = model(x, xyz, category)
```


## Reference

```bibtex
@InProceedings{qian2022pointnext,
  title   = {PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies},
  author  = {Qian, Guocheng and Li, Yuchen and Peng, Houwen and Mai, Jinjie and Hammoud, Hasan and Elhoseiny, Mohamed and Ghanem, Bernard},
  booktitle=Advances in Neural Information Processing Systems (NeurIPS),
  year    = {2022},
}
```

