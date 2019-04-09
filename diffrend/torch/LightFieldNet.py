import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential


class LFNetBase(nn.Module):
    """Light Field Network
    Incident illumination at a vertex in a given direction
    """
    def __init__(self, in_ch, out_ch, **params):
        super(LFNetBase, self).__init__()
        self.in_ch = in_ch  # 3 position + 3 direction
        self.out_ch = out_ch  # 3 RGB radiance
        self.chunks = 4 if 'chunks' not in params else params['chunks']
        self.params = params

    def forward(self, x):
        modules = [module for k, module in self._modules.items()][0]
        x = checkpoint_sequential(modules, self.chunks, x)
        return x


class LFNetV0(LFNetBase):
    def __init__(self, **params):
        super(LFNetV0, self).__init__(**params)
        self.net = nn.Sequential(
            nn.Linear(self.in_ch, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.out_ch),
            nn.ReLU()
        )


def test_LFNet():
    from diffrend.torch.utils import tch_var_f
    import numpy as np
    pos = tch_var_f(list(np.random.rand(1, 10, 8)))
    y = LFNetV0(in_ch=8, out_ch=3).cuda()(pos)
    print(y)
    print(y.shape, y)


if __name__ == '__main__':
    test_LFNet()
