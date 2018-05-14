import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffrend.torch.utils import tch_var_f
from diffrend.torch.ops import sph2cart_unit


class NEstNet(nn.Module):
    """Normal Estimation Network
    Estimates surface normals from vertex positions
    """
    def __init__(self, sph=True):
        super(NEstNet, self).__init__()
        self.sph_out = sph
        out_ch = 2 if sph else 3
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.Conv2d(64, out_ch, 3, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.net(x)
        if self.sph_out:
            x = F.sigmoid(x) * tch_var_f([2 * np.pi, np.pi / 2])[np.newaxis, :, np.newaxis, np.newaxis]
            x = sph2cart_unit(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            sum_squared = torch.sum(x ** 2, dim=1)
            x = x / torch.sqrt(sum_squared + 1e-12)
        return x


def test_NEstNet():
    import numpy as np
    pos = torch.autograd.Variable(torch.FloatTensor(np.random.rand(1, 3, 5, 5)))
    y = NEstNet(sph=False)(pos)
    print(y.shape, y.norm(dim=1))
    y = NEstNet()(pos)
    print(y.shape, y.norm(dim=1))


if __name__ == '__main__':
    print(NEstNet())
    test_NEstNet()
