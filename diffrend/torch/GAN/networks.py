"""Networks."""
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def create_networks(opt, verbose=True):
    """Create the networks."""
    # Parameters
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    # render_img_nc = int(opt.render_img_nc)
    # splats_img_size = int(opt.splats_img_size)
    n_splats = int(opt.n_splats)
    # render_img_size = int(opt.width)
    splats_n_dims = 6

    # Create generator network
    if opt.gen_type == 'mlp':
        netG = _netG_mlp(ngpu, nz, ngf, splats_n_dims, n_splats)
    elif opt.gen_type == 'resnet':
        netG = _netG_resnet(nz, splats_n_dims, n_splats)
    elif opt.gen_type == 'cnn':
        netG = _netG(ngpu, nz, ngf, splats_n_dims, use_tanh=False,
                     bias_type=opt.gen_bias_type)
    else:
        raise ValueError("Unknown generator")

    # Init weights/load pretrained model
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    else:
        netG.apply(weights_init)

    # Create the discriminator network
    if opt.criterion == 'WGAN':
        netD = _netD(ngpu, 1, ndf, 1)
    else:
        netD = _netD(ngpu, 1, ndf)

    # Init weights/load pretrained model
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    else:
        netD.apply(weights_init)

    # Show networks
    if verbose:
        print(netG)
        print(netD)

    return netG, netD


# Custom weights initialization called on netG and netD
def weights_init(m):
    """Weight initializer."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


#############################################
# Modules for conditional batchnorm
#############################################
class TwoInputModule(nn.Module):
    """Abstract class."""

    def forward(self, input1, input2):
        """Forward method."""
        raise NotImplementedError


class CondBatchNorm(nn.BatchNorm2d, TwoInputModule):
    """Conditional batch norm."""

    def __init__(self, x_dim, z_dim, eps=1e-5, momentum=0.1):
        """Constructor.

        - `x_dim`: dimensionality of x input
        - `z_dim`: dimensionality of z latents
        """
        super(CondBatchNorm, self).__init__(x_dim, eps, momentum, affine=False)
        self.eps = eps
        self.shift_conv = nn.Sequential(
            nn.Conv2d(z_dim, x_dim, kernel_size=1, padding=0, bias=True),
            # nn.ReLU(True)
        )
        self.scale_conv = nn.Sequential(
            nn.Conv2d(z_dim, x_dim, kernel_size=1, padding=0, bias=True),
            # nn.ReLU(True)
        )

    def forward(self, input, noise):
        """Forward method."""
        shift = self.shift_conv.forward(noise)
        scale = self.scale_conv.forward(noise)

        norm_features = super(CondBatchNorm, self).forward(input)
        output = norm_features * scale + shift
        return output


#############################################
# Generators
#############################################
class _netG(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, use_tanh=False, bias_type=None):
        super(_netG, self).__init__()
        # Save parameters
        self.ngpu = ngpu
        self.nc = nc
        self.use_tanh = use_tanh
        self.bias_type = bias_type

        # Main layers
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, nc, 4, 2, 1, bias=False),
            # state size. (ngf) x 32 x 32
        )

        # Coordinates bias
        if bias_type == 'plane':
            coords_tmp = np.array(list(np.ndindex((32, 32)))).reshape(2, 32,
                                                                      32)
            coords = np.zeros((1, nc, 32, 32), dtype=np.float32)
            coords[0, :2, :, :] = coords_tmp/32.
            self.coords = Variable(torch.FloatTensor(coords))
            if torch.cuda.is_available():
                self.coords = self.coords.cuda()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            out = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            # Generate the output
            out = self.main(input)
            if self.use_tanh:
                out = nn.Tanh(out)

            # Add bias to enforce locality
            if self.bias_type is not None:
                coords = self.coords.expand(out.size()[0], self.nc, 32, 32)
                out = out + coords

            # Reshape output
            out = out.view(out.size(0), self.nc, -1)
            out = out.permute(0, 2, 1)

        return out


class _netG_mlp(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, nsplats):
        super(_netG_mlp, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.nspalts = nsplats
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(nz, ngf*4),
            # nn.BatchNorm1d(ngf*4),
            nn.LeakyReLU(0.2, True),

            nn.Linear(ngf*4, ngf*16),
            nn.BatchNorm1d(ngf*16),
            nn.LeakyReLU(0.2, True),

            nn.Linear(ngf*16, ngf*16),
            nn.BatchNorm1d(ngf*16),
            nn.LeakyReLU(0.2, True),

            nn.Linear(ngf*16, ngf*32),
            # nn.BatchNorm1d(ngf*16),
            nn.LeakyReLU(0.2, True),

            nn.Linear(ngf*32, ngf*64),
            nn.LeakyReLU(0.2, True),

            nn.Linear(ngf*64, nc*nsplats)
            # nn.BatchNorm1d(ndf*4),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(ndf*4, 1)
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            input = input.view(input.size()[0], input.size()[1])
            output = self.main(input)
            output = output.view(output.size(0), self.nspalts, self.nc)
        return output


class ResBlock(nn.Module):
    """Resnet block."""

    def __init__(self, dim=512, res_weight=0.3):
        """Constructor."""
        super(ResBlock, self).__init__()
        self.res_weight = res_weight
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(dim, dim, 3, padding=1),
        )

    def forward(self, input):
        """Forward method."""
        output = self.res_block(input)
        return input + (self.res_weight*output)


class _netG_resnet(nn.Module):
    def __init__(self, nz, nc, nsplats, dim=512, res_weight=0.3):
        super(_netG_resnet, self).__init__()
        self.dim = dim
        self.nc = nc
        self.fc1 = nn.Linear(nz, dim*nc)
        self.block = nn.Sequential(
            ResBlock(dim, res_weight),
            ResBlock(dim, res_weight),
            ResBlock(dim, res_weight),
            ResBlock(dim, res_weight),
            ResBlock(dim, res_weight),
            nn.Conv1d(dim, nsplats, 3, padding=1),
            nn.BatchNorm1d(nsplats),
            nn.Conv1d(nsplats, nsplats, 1),
        )

    def forward(self, noise):
        noise = noise.view(noise.size()[0], noise.size()[1])
        output = self.fc1(noise)
        output = output.view(-1, self.dim, self.nc)
        output = self.block(output)
        return output


#############################################
# Discriminators
#############################################
class _netD(nn.Module):
    def __init__(self, ngpu, nc, ndf, no_sigmoid=0):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.no_sigmoid = no_sigmoid

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=True),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 1).squeeze(1)

        if self.no_sigmoid == 1:
            return x
        else:
            return F.sigmoid(x)
