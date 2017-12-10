"""Parameters module."""
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data


class Parameters():
    """base options."""

    def __init__(self):
        """Constructor."""
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        """Initialize."""

        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
        self.parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
        self.parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
        self.parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--ndf', type=int, default=64)
        self.parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
        self.parser.add_argument('--no_cuda', action='store_true', default=False, help='enables cuda')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--netG', default='', help="path to netG (to continue training)")
        self.parser.add_argument('--netD', default='', help="path to netD (to continue training)")
        self.parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
        self.parser.add_argument('--manualSeed', type=int, help='manual seed')

    def parse(self):
        """Parse."""
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        print(self.opt)

        # Make output folder
        try:
            os.makedirs(self.opt.outf)
        except OSError:
            pass

        # Set random seed
        if self.opt.manualSeed is None:
            self.opt.manualSeed = random.randint(1, 10000)
        print("Random Seed: ", self.opt.manualSeed)
        random.seed(self.opt.manualSeed)
        torch.manual_seed(self.opt.manualSeed)
        if not self.opt.no_cuda:
            torch.cuda.manual_seed_all(self.opt.manualSeed)

        # Check CUDA is selected
        cudnn.benchmark = True
        if torch.cuda.is_available() and self.opt.no_cuda:
            print("WARNING: You have a CUDA device, so you should "
                  "probably run with --cuda")

        return self.opt
