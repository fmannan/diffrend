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

    def __init__(self,DIR_DATA):
        """Constructor."""
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.DIR_DATA = DIR_DATA

    def initialize(self):
        """Initialize."""
        # parser = argparse.ArgumentParser(usage="splat_gen_render_demo.py --model filename --out_dir output_dir "
        #                                        "--n 5000 --width 128 --height 128 --r 0.025 --cam_dist 5 --nv 10")
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
        self.parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
        self.parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
        self.parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--ndf', type=int, default=64)
        self.parser.add_argument('--niter', type=int, default=4000, help='number of epochs to train for')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
        self.parser.add_argument('--toward_cam', type=float, default=1.0, help='weight to towards_cam_loss')
        self.parser.add_argument('--im_grad', type=float, default=1.0, help='weight to img_gradientloss')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
        self.parser.add_argument('--no_cuda', action='store_true', default=False, help='enables cuda')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--netG', default='', help="path to netG (to continue training)")
        self.parser.add_argument('--netD', default='', help="path to netD (to continue training)")
        self.parser.add_argument('--manualSeed', type=int, help='manual seed')
        self.parser.add_argument('--model', type=str, default=self.DIR_DATA + '/chair_0001.off')
        self.parser.add_argument('--out_dir', type=str, default='/data/lisa/data/sai/renderer_chair_64_2K_75_nogp_imggen_strongdis')
        # self.parser.add_argument('--out_dir', type=str, default='/data/lisa/data/sai/renderer_bunny_64_sameview_check_separatefake')
        self.parser.add_argument('--gen_type', type=str, default='mlp')
        self.parser.add_argument('--dis_type', type=str, default='resnet')
        self.parser.add_argument('--width', type=int, default=64)
        self.parser.add_argument('--height', type=int, default=64)
        self.parser.add_argument('--n', type=int, default=2000)
        self.parser.add_argument('--r', type=float, default=0.03)
        self.parser.add_argument('--cam_dist', type=float, default=5.0, help='Camera distance from the center of the object')
        self.parser.add_argument('--nv', type=int, default=10, help='Number of views to generate')
        self.parser.add_argument('--fovy', type=float, default=15.0, help='Field of view in the vertical direction')
        self.parser.add_argument('--f', type=float, default=0.1, help='focal length')
        self.parser.add_argument('--same_view', action='store_true', default=False, help='data with view fixed')
        self.parser.add_argument("--criterion", help="GAN Training criterion", choices=['GAN', 'WGAN'], default='WGAN')
        self.parser.add_argument("--gp", help="Add gradient penalty", choices=['None', 'original'], default='original')
        self.parser.add_argument("--gp_lambda", help="GP lambda", type=float, default=10.)


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

        # Check CUDA is selected
        cudnn.benchmark = True
        if torch.cuda.is_available() and self.opt.no_cuda:
            print("WARNING: You have a CUDA device, so you should "
                  "probably run with --cuda")

        return self.opt
