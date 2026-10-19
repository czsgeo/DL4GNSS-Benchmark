"""tcn.py -- dilated temporal-convolutional network for the benchmark tasks.

A third learned family, materially different from the U-Net (pooling-based
encoder/decoder) and the BiGRU (recurrence): stacked residual blocks of dilated
causal-free convolutions. The kernel is centred (non-causal) because all three
tasks are offline; dilation grows geometrically so the receptive field covers
several hundred days without a deep stack.

Parameter budget is matched to the U-Net of `dl.build_unet(base=12, depth=3)`
(~2.13e5) so architecture comparisons are not confounded by capacity.
"""
from __future__ import annotations

import math


class FastConv1d:
    """Drop-in Conv1d (groups=1) built from unfold + matmul.

    The cuDNN conv path is unavailable on the machine this benchmark was run on,
    and the native CUDA fallback is ~19x slower than this implementation for the
    lengths used here (1.14 s vs 0.059 s per 32 x 1024 forward/backward step).
    The result is the same convolution to float32 rounding (max |diff| 5e-8 on
    the trained shapes), so checkpoints are interchangeable with nn.Conv1d.
    """

    def __new__(cls, cin, cout, k, padding=0, dilation=1, bias=True):
        import torch
        import torch.nn as nn

        class _FastConv1d(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(cout, cin, k))
                if bias:
                    self.bias = nn.Parameter(torch.empty(cout))
                else:
                    self.register_parameter("bias", None)
                self.padding = padding
                self.dilation = dilation
                self.k = k
                self.reset_parameters()

            def reset_parameters(self):
                nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                if self.bias is not None:
                    bound = 1.0 / math.sqrt(self.weight.shape[1] * self.k)
                    nn.init.uniform_(self.bias, -bound, bound)

            def forward(self, x):
                import torch.nn.functional as F

                if self.padding:
                    x = F.pad(x, (self.padding, self.padding))
                if self.dilation == 1:
                    u = x.unfold(2, self.k, 1)
                else:
                    span = self.dilation * (self.k - 1) + 1
                    u = x.unfold(2, span, 1)[..., ::self.dilation]
                b, c, l, k = u.shape
                u = u.permute(0, 2, 1, 3).reshape(b, l, c * k)
                out = u @ self.weight.reshape(self.weight.shape[0], -1).t()
                out = out.transpose(1, 2)
                if self.bias is not None:
                    out = out + self.bias[None, :, None]
                return out

        return _FastConv1d()


def build_tcn(in_ch=1, channels=(40, 56, 72, 80, 80), kernel=5,
              dilations=(1, 2, 4, 8, 16)):
    """Return an nn.Module mapping (B, in_ch, W) -> (B, 1, W)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    assert len(channels) == len(dilations)

    class ResBlock(nn.Module):
        def __init__(self, cin, cout, k, d):
            super().__init__()
            pad = d * (k - 1) // 2
            self.c1 = FastConv1d(cin, cout, k, padding=pad, dilation=d)
            self.c2 = FastConv1d(cout, cout, k, padding=pad, dilation=d)
            self.skip = (FastConv1d(cin, cout, 1) if cin != cout else nn.Identity())

        def forward(self, x):
            z = F.relu(self.c1(x))
            z = self.c2(z)
            return F.relu(z + self.skip(x))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = FastConv1d(in_ch, channels[0], 7, padding=3)
            blocks = []
            cin = channels[0]
            for cout, d in zip(channels, dilations):
                blocks.append(ResBlock(cin, cout, kernel, d))
                cin = cout
            self.blocks = nn.Sequential(*blocks)
            self.head = FastConv1d(cin, 1, 7, padding=3)

        def forward(self, x):
            x = torch.relu(self.stem(x))
            x = self.blocks(x)
            return self.head(x)

    return Net()


def count_params(net):
    return int(sum(p.numel() for p in net.parameters()))
