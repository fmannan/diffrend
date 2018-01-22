"""Parameters module."""
import argparse
import os
import random
import getpass
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
        # Define training set depending on the user name
        username = getpass.getuser()
        if username == 'dvazquez' or username == 'root':
            default_root = '/home/dvazquez/datasets/shapenet/ShapeNetCore.v2'
            # default_root = '/mnt/AIDATA/home/dvazquez/datasets/shapenet/ShapeNetCore.v2'
            default_out = './render_samples/'
        elif username == 'florian':
            default_root = '/data/lisa/data/ShapeNetCore.v2'
            # default_root = '/media/florian/8BAA-82D3/shapenet'
            default_out = './render_samples/'
        elif username == 'fahim':
            default_root = '/data/lisa/data/ShapeNetCore.v2'
            default_out = './render_samples/'
        elif username == 'sai':
            default_root = '/data/lisa/data/ShapeNetCore.v2'
            default_out = './render_samples/'
            # default_out = '/data/lisa/data/sai/renderer_bunny_64_sameview_check_separatefake'
        else:
            raise ValueError('Add the route for the dataset of your system')

        # Dataset parameters
        self.parser.add_argument('--dataset', type=str, default='shapenet', help='dataset name')
        self.parser.add_argument('--root_dir', type=str, default=default_root, help='dataset root directory')
        self.parser.add_argument('--synsets', type=str, default='', help='Synsets from the shapenet dataset to use')
        self.parser.add_argument('--classes', type=str, default='airplane,microphone', help='Classes from the shapenet dataset to use')
        self.parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
        # corresponding folders: 02691156, 03759954

        # other low-footprint objects:
        # 02773838 - bag, 83
        # 02801938 - basket, 113
        # 02880940 - bowl, 186
        # 02942699 - camera, 113
        # 03261776 - headphone, 73
        # 03513137 - helmet, 162
        # 03797390 - mug, 214
        # 04004475 - printer, 166
        # bowl,mug

        # Network parameters
        self.parser.add_argument('--gen_type', type=str, default='cnn')
        self.parser.add_argument('--gen_bias_type', type=str, default=None)
        self.parser.add_argument('--netG', default='', help="path to netG (to continue training)")
        self.parser.add_argument('--netD', default='', help="path to netD (to continue training)")
        self.parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
        self.parser.add_argument('--ngf', type=int, default=64, help='number of features in the generator network')
        self.parser.add_argument('--ndf', type=int, default=64, help='number of features in the discriminator network')

        # Optimization parameters
        self.parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer (adam, rmsprop)')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
        self.parser.add_argument('--n_iter', type=int, default=4000, help='number of iterations to train')
        self.parser.add_argument('--batchSize', type=int, default=8, help='input batch size')

        # GAN parameters
        self.parser.add_argument("--criterion", help="GAN Training criterion", choices=['GAN', 'WGAN'], default='WGAN')
        self.parser.add_argument("--gp", help="Add gradient penalty", choices=['None', 'original'], default='original')
        self.parser.add_argument("--gp_lambda", help="GP lambda", type=float, default=10.)
        self.parser.add_argument("--critic_iters", type=int, default=5, help="Number of critic iterations")
        self.parser.add_argument('--clamp', type=float, default=0.01, help='clamp the weights for WGAN')

        # Other parameters
        self.parser.add_argument('--no_cuda', action='store_true', default=False, help='enables cuda')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--manualSeed', type=int, help='manual seed')
        self.parser.add_argument('--out_dir', type=str, default=default_out)

        # Camera parameters
        self.parser.add_argument('--width', type=int, default=64)
        self.parser.add_argument('--height', type=int, default=64)
        self.parser.add_argument('--cam_dist', type=float, default=5.0, help='Camera distance from the center of the object')
        self.parser.add_argument('--nv', type=int, default=10, help='Number of views to generate')
        self.parser.add_argument('--fovy', type=float, default=15.0, help='Field of view in the vertical direction')
        self.parser.add_argument('--focal_length', type=float, default=0.1, help='focal length')

        # Rendering parameters
        self.parser.add_argument('--splats_img_size', type=int, default=32, help='the height / width of the number of generator splats')
        self.parser.add_argument('--render_img_nc', type=int, default=1, help='Number of channels of the render image')
        self.parser.add_argument('--render_img_size', type=int, default=64, help='Width/height of the rendering image')
        self.parser.add_argument('--splats_radius', type=float, default=0.025, help='radius of the splats (fix)')
        self.parser.add_argument('--same_view', action='store_true', default=True, help='data with view fixed')

    def parse(self):
        """Parse."""
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        print(self.opt)

        # Make output folder
        try:
            os.makedirs(self.opt.out_dir)
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

        # Set number of splats param
        self.opt.n_splats = self.opt.splats_img_size*self.opt.splats_img_size

        # Check CUDA is selected
        cudnn.benchmark = True
        if torch.cuda.is_available() and self.opt.no_cuda:
            print("WARNING: You have a CUDA device, so you should "
                  "probably run with --cuda")

        return self.opt
