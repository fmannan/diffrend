import torch
import torch.nn as nn


class NEstNet(nn.Module):
    """Normal Estimation Network
    Estimates surface normals from vertex positions
    """
    def __init__(self):
        super(NEstNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.Conv2d(64, 3, 3, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.net(x)
        sum_squared = torch.sum(x ** 2, dim=1)
        return x / torch.sqrt(sum_squared + 1e-12)


def test_NEstNet():
    import numpy as np
    pos = torch.autograd.Variable(torch.FloatTensor(np.random.rand(1, 3, 5, 5)))
    y = NEstNet()(pos)
    print(y.shape, y.norm(dim=1))


if __name__ == '__main__':
    print(NEstNet())
    test_NEstNet()
