import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffrend.torch.utils import tch_var_f
from diffrend.torch.ops import sph2cart_unit


class NEstNetBase(nn.Module):
    """Normal Estimation Network
    Estimates surface normals from vertex positions
    """
    def __init__(self, in_channel=3, sph=True):
        super(NEstNetBase, self).__init__()
        self.sph_out = sph
        self.in_ch = in_channel
        self.out_ch = 2 if sph else 3

    def forward(self, x):
        x = self.net(x)
        if self.sph_out:
            x = F.sigmoid(x) * tch_var_f([2 * np.pi, np.pi / 2])[np.newaxis, :, np.newaxis, np.newaxis]
            x = sph2cart_unit(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            x = torch.cat([x[:, 0, :, :][:, np.newaxis, ...],
                           x[:, 1, :, :][:, np.newaxis, ...],
                           torch.abs(x[:, 2, :, :][:, np.newaxis, ...])], dim=1)
            sum_squared = torch.sum(x ** 2, dim=1)
            x = x / torch.sqrt(sum_squared + 1e-12)
        return x


class NEstNetV0(NEstNetBase):
    def __init__(self, sph=True):
        super(NEstNetV0, self).__init__(sph=sph)
        self.net = nn.Sequential(
            nn.Conv2d(self.in_ch, 64, 3, padding=1, bias=False),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.Conv2d(64, self.out_ch, 3, padding=1, bias=False),
        )


class NEstNetV1(NEstNetBase):
    def __init__(self, sph=True):
        super(NEstNetV1, self).__init__(sph=sph)
        self.net = nn.Sequential(
            nn.Conv2d(self.in_ch, 64, 3, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(64, self.out_ch, 3, padding=1, bias=False),
        )


class NEstNetV1_2(NEstNetBase):
    def __init__(self, sph=True):
        super(NEstNetV1_2, self).__init__(sph=sph)
        self.net = nn.Sequential(
            # Downsample
            nn.Conv2d(self.in_ch, 64, 3, padding=0, bias=False),
            nn.PReLU(),
            nn.Conv2d(64, 128, 3, padding=0, bias=False),
            nn.PReLU(),
            nn.Conv2d(128, 256, 3, padding=0, bias=False),
            nn.PReLU(),
            # Upsample
            nn.ConvTranspose2d(256, 128, 3),
            nn.PReLU(),
            nn.ConvTranspose2d(128, 64, 3),
            nn.PReLU(),
            nn.ConvTranspose2d(64, 64, 3),
            nn.PReLU(),
            # Final
            nn.Conv2d(64, self.out_ch, 3, padding=1, bias=False),
        )


class NEstNetAffine(NEstNetBase):
    def __init__(self, kernel_size, sph=True):
        super(NEstNetAffine, self).__init__(sph=sph)
        self.net = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False),
        )


def test_NEstNet():
    import numpy as np
    pos = tch_var_f(list(np.random.rand(1, 3, 5, 5)))
    y = NEstNetV0(sph=False).cuda()(pos)
    print(y.shape, y.norm(dim=1))
    y = NEstNetAffine(kernel_size=3).cuda()(pos)
    print(y.shape, y.norm(dim=1))


if __name__ == '__main__':
    test_NEstNet()
