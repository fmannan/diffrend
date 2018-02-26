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
            # default_root = '/mnt/home/dvazquez/datasets/shapenet/ShapeNetCore.v2'
            default_out = './render_samples/'
        elif username == 'florian':
            default_root = '/lindata/datasets/shapenet/ShapeNetCore.v2'
            # default_root = '/data/lisa/data/ShapeNetCore.v2'
            # default_root = '/media/florian/8BAA-82D3/shapenet'
            default_out = './render_samples/'
        elif username == 'fahim' or username == 'fmannan':
            default_root = '/data/lisa/data/ShapeNetCore.v2'
            default_out = './render_samples/'
        elif username == 'mudumbas':
            default_root = '/data/lisa/data/ShapeNetCore.v2'
            default_out = '/data/lisa/data/sai'
            # default_out = '/data/lisa/data/sai/renderer_bunny_64_sameview_check_separatefake'
        else:
            raise ValueError('Add the route for the dataset of your system')

        # Dataset parameters
        self.parser.add_argument('--dataset', type=str, default='shapenet', help='dataset name')
        self.parser.add_argument('--root_dir', type=str, default=default_root, help='dataset root directory')
        self.parser.add_argument('--synsets', type=str, default='', help='Synsets from the shapenet dataset to use')
        self.parser.add_argument('--classes', type=str, default='bowl', help='Classes from the shapenet dataset to use')
        self.parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
        self.parser.add_argument('--toy_example', action='store_true', default=False, help='Use toy example')
        self.parser.add_argument('--use_mesh', action='store_true', default=True, help='Render dataset with meshes')
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
        self.parser.add_argument('--gen_type', type=str, default='dcgan', help='One of: mlp, cnn, dcgan, resnet') # try resnet :)
        self.parser.add_argument('--gen_norm', type=str, default='batchnorm', help='One of: None, batchnorm, instancenorm')
        self.parser.add_argument('--ngf', type=int, default=90, help='number of features in the generator network')
        self.parser.add_argument('--gen_nextra_layers', type=int, default=0, help='number of extra layers in the generator network')
        self.parser.add_argument('--gen_bias_type', type=str, default=None, help='One of: None, plane')
        self.parser.add_argument('--netG', default='', help="path to netG (to continue training)")
        self.parser.add_argument('--fix_splat_pos', action='store_true', default=True, help='X and Y coordinates are fix')
        self.parser.add_argument('--norm_sph_coord', action='store_true', default=True, help='Use spherical coordinates for the normal')
        self.parser.add_argument('--max_gnorm', type=float, default=400., help='max grad norm to which it will be clipped (if exceeded)')
        self.parser.add_argument('--disc_type', type=str, default='dcgan', help='One of: cnn, dcgan')
        self.parser.add_argument('--disc_norm', type=str, default='None', help='One of: None, batchnorm, instancenorm')
        self.parser.add_argument('--ndf', type=int, default=64, help='number of features in the discriminator network')
        self.parser.add_argument('--disc_nextra_layers', type=int, default=0, help='number of extra layers in the discriminator network')
        self.parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
        self.parser.add_argument('--netD', default='', help="path to netD (to continue training)")

        # Optimization parameters
        self.parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer (adam, rmsprop)')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
        self.parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for adam. default=0.5')
        self.parser.add_argument('--n_iter', type=int, default=40000, help='number of iterations to train')
        self.parser.add_argument('--batchSize', type=int, default=2, help='input batch size')

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
        self.parser.add_argument('--name', type=str, default='',required=False)

        # Camera parameters
        self.parser.add_argument('--width', type=int, default=128)
        self.parser.add_argument('--height', type=int, default=128)
        self.parser.add_argument('--cam_dist', type=float, default=4.0, help='Camera distance from the center of the object')
        self.parser.add_argument('--nv', type=int, default=10, help='Number of views to generate')
        self.parser.add_argument('--angle', type=int, default=5, help='cam angle')
        self.parser.add_argument('--fovy', type=float, default=20, help='Field of view in the vertical direction. Default: 15.0')
        self.parser.add_argument('--focal_length', type=float, default=0.1, help='focal length')
        self.parser.add_argument('--theta', nargs=2, type=float, default=None, help='Angle in degrees from the z-axis.')
        self.parser.add_argument('--phi', nargs=2, type=float, default=None, help='Angle in degrees from the x-axis.')
        self.parser.add_argument('--axis', nargs=3, type=float, default=[2, 1, 2],help='Axis for random camera position.')
        self.parser.add_argument('--cam_pos', nargs=3, type=float, help='Camera position.')
        self.parser.add_argument('--at', nargs=3, default=[0, 0, 0], type=float, help='Camera lookat position.')
        self.parser.add_argument('--sphere-halfbox', action='store_true', help='Renders demo sphere-halfbox')
        self.parser.add_argument('--norm_depth_image_only', action='store_true', default=False, help='Render on the normalized'
                                                                                            ' depth image.')
        self.parser.add_argument('--test_cam_dist', action='store_true', help='Check if the images are consistent with a'
                                                                     'camera at a fixed distance.')

        # Rendering parameters
        self.parser.add_argument('--splats_img_size', type=int, default=128, help='the height / width of the number of generator splats')
        self.parser.add_argument('--render_type', type=str, default='img', help='render the image or the depth map [img, depth]')
        self.parser.add_argument('--render_img_size', type=int, default=128, help='Width/height of the rendering image')
        self.parser.add_argument('--splats_radius', type=float, default=0.05, help='radius of the splats (fix)')
        self.parser.add_argument('--same_view', action='store_true', help='data with view fixed') # before we add conditioning on cam pose, this is necessary

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

        # Set render number of channels
        if self.opt.render_type == 'img':
            self.opt.render_img_nc = 3
        elif self.opt.render_type == 'depth':
            self.opt.render_img_nc = 1
        else:
            raise ValueError('Unknown rendering type')

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
