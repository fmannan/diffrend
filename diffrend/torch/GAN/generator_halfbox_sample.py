"""Conditional Generator only on cam_pos."""
from __future__ import absolute_import

import copy
import numpy as np
from scipy.misc import imsave
import os
import sys
#sys.path.append('../../..')
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import shutil
import torchvision
from diffrend.torch.GAN.datasets import Dataset_load
from diffrend.torch.GAN.cnetworks import create_networks
from diffrend.torch.GAN.parameters_halfbox import Parameters
from diffrend.torch.GAN.utils import make_dot
from diffrend.torch.params import SCENE_BASIC,  SCENE_SPHERE_HALFBOX
from diffrend.torch.utils import tch_var_f, tch_var_l, where, get_data, normalize, cam_to_world
from diffrend.torch.renderer import render, render_splats_NDC, render_splats_along_ray
from diffrend.utils.sample_generator import uniform_sample_sphere
from diffrend.torch.ops import sph2cart_unit
# try: # temporarily
#     from hyperdash import Experiment
#     HYPERDASH_SUPPORTED = True
# except ImportError:
HYPERDASH_SUPPORTED = False
def copy_scripts_to_folder(expr_dir):
    shutil.copy("cnetworks.py", expr_dir)
    shutil.copy("parameters_halfbox.py", expr_dir)
    shutil.copy(__file__, expr_dir)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def create_scene(width, height, fovy, focal_length, n_samples,
                 ):
    """Create a semi-empty scene with camera parameters."""
    # Create a splats rendering scene
    scene = copy.deepcopy(SCENE_SPHERE_HALFBOX)

    # Define the camera parameters
    scene['camera']['viewport'] = [0, 0, width, height]
    scene['camera']['fovy'] = np.deg2rad(fovy)
    scene['camera']['focal_length'] = focal_length

    return scene


def calc_gradient_penalty(discriminator, real_data, fake_data, fake_data_cond,gp_lambda):
    """Calculate GP."""
    assert real_data.size(0) == fake_data.size(0)
    alpha = torch.rand(real_data.size(0), 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = Variable(alpha * real_data + ((1 - alpha) * fake_data),
                            requires_grad=True)
    interpolates_cond =  Variable(fake_data_cond, requires_grad=True)

    disc_interpolates = discriminator(interpolates, interpolates_cond)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda

    return gradient_penalty


class GAN(object):
    """GAN class."""

    def __init__(self, opt, dataset_load=None, experiment=None, exp_dir=None):
        """Constructor."""
        self.opt = opt
        self.exp = experiment
        self.real_label = 1
        self.fake_label = 0
        self.dataset_load = dataset_load
        self.opt.out_dir=exp_dir
        # Create dataset loader
        self.create_dataset_loader()

        # Create the networks
        self.create_networks()

        # Create create_tensors
        self.create_tensors()

        # Create criterion
        self.create_criterion()

        # Create create optimizers
        self.create_optimizers()

        # Create splats rendering scene
        self.create_scene()

    def create_dataset_loader(self, ):
        """Create dataset leader."""
        # Define camera positions
        if self.opt.same_view:
            if not self.opt.toy_example:
                self.cam_pos = uniform_sample_sphere(radius=self.opt.cam_dist,
                                                     num_samples=1)
            else:
                arrays = [np.asarray([3., 3., 3.]) for _ in range(self.opt.batchSize)]
                self.cam_pos = np.stack(arrays, axis=0)

        # Create dataset loader
        #self.dataset_load.initialize_dataset()
        #self.dataset = self.dataset_load.get_dataset()
        #self.dataset_load.initialize_dataset_loader()
        #self.dataset_loader = self.dataset_load.get_dataset_loader()

    def create_networks(self, ):
        """Create networks."""
        self.netG, self.netD = create_networks(self.opt, verbose=True)
        if not self.opt.no_cuda:
            self.netD = self.netD.cuda()
            self.netG = self.netG.cuda()

    def create_scene(self, ):
        """Create a semi-empty scene with camera parameters."""
        self.scene = create_scene(
            self.opt.width, self.opt.height, self.opt.fovy,
            self.opt.focal_length, self.opt.n_splats)

    def create_tensors(self, ):
        """Create the tensors."""
        self.input = torch.FloatTensor(
            self.opt.batchSize, self.opt.render_img_nc,
            self.opt.render_img_size, self.opt.render_img_size)
        self.input_cond = torch.FloatTensor(
            self.opt.batchSize, 3)
        self.noise = torch.FloatTensor(
            self.opt.batchSize, int(self.opt.nz), 1, 1)
        self.fixed_noise = torch.FloatTensor(
            self.opt.batchSize, int(self.opt.nz), 1, 1).normal_(0, 1)
        self.label = torch.FloatTensor(self.opt.batchSize)
        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1

        if not self.opt.no_cuda:
            self.input = self.input.cuda()
            self.input_cond = self.input_cond.cuda()
            self.label = self.label.cuda()
            self.noise = self.noise.cuda()
            self.fixed_noise = self.fixed_noise.cuda()
            self.one = self.one.cuda()
            self.mone = self.mone.cuda()

        self.fixed_noise = Variable(self.fixed_noise)

    def create_criterion(self, ):
        """Create criterion."""
        self.criterion = nn.BCELoss()
        if not self.opt.no_cuda:
            self.criterion = self.criterion.cuda()

    def create_optimizers(self, ):
        """Create optimizers."""

        # model_parameters = filter(lambda p: p.requires_grad, self.netG.parameters())
        # params = sum([np.prod(p.size()) for p in model_parameters])
        # print("trainable parameters:",params)

        if self.opt.optimizer == 'adam':
            self.optimizerD = optim.Adam(self.netD.parameters(),
                                         lr=self.opt.lr,
                                         betas=(self.opt.beta1, 0.999))
            self.optimizerG = optim.Adam(self.netG.parameters(),
                                         lr=self.opt.lr,
                                         betas=(self.opt.beta1, 0.999))
        elif self.opt.optimizer == 'rmsprop':
            self.optimizerD = optim.RMSprop(self.netD.parameters(),
                                            lr=self.opt.lr)
            self.optimizerG = optim.RMSprop(self.netG.parameters(),
                                            lr=self.opt.lr)
        else:
            raise ValueError('Unknown optimizer: ' + self.opt.optimizer)


    def get_real_samples(self):
        """Get a real sample."""
        if not self.opt.toy_example:
            # Load a batch of samples
            try:
                samples = self.data_iter.next()
            except StopIteration:
                del self.data_iter
                self.data_iter = iter(self.dataset_loader)
                samples = self.data_iter.next()
            except AttributeError:
                self.data_iter = iter(self.dataset_loader)
                samples = self.data_iter.next()
        # else:
        #     samples = self.generate_toy_samples()

        # Define the camera poses
        if not self.opt.same_view:
            self.cam_pos = uniform_sample_sphere(radius=self.opt.cam_dist, num_samples=self.opt.batchSize,
                                                 axis=self.opt.axis, angle=np.deg2rad(self.opt.angle),
                                                 theta_range=self.opt.theta, phi_range=self.opt.phi)

        # Create a splats rendering scene
        large_scene = create_scene(self.opt.width, self.opt.height,
                                   self.opt.fovy, self.opt.focal_length,
                                   self.opt.n_splats)
        lookat = self.opt.at if self.opt.at is not None else [0.0, 0.0, 0.0, 1.0]
        large_scene['camera']['at'] = tch_var_f(lookat)

        # Render scenes
        #import ipdb; ipdb.set_trace()
        data = []
        data_cond=[]
        for idx in range(self.opt.batchSize):

            # Set camera position
            if not self.opt.same_view:
                large_scene['camera']['eye'] = tch_var_f(self.cam_pos[idx])
            else:
                large_scene['camera']['eye'] = tch_var_f(self.cam_pos[0])
            view_dir = normalize(large_scene['camera']['at'] - large_scene['camera']['eye'])
            # Render scene
            res = render(large_scene,norm_depth_image_only=self.opt.norm_depth_image_only)



            # Get rendered output
            if self.opt.render_img_nc == 1:
                depth = res['depth']
                # Normalize depth image
                cond = depth >= large_scene['camera']['far']
                depth = where(cond, torch.min(depth), depth)
                im = ((depth - torch.min(depth)) /
                      (torch.max(depth) - torch.min(depth)))
                im = im.unsqueeze(0)
            else:
                im = res['image'].permute(2, 0, 1)

            # Add depth image to the output structure
            data.append(im)
            #import ipdb; ipdb.set_trace()
            data_cond.append(large_scene['camera']['eye'])
        # Stack real samples
        real_samples = torch.stack(data)
        real_samples_cond = torch.stack(data_cond)
        self.batch_size = real_samples.size(0)
        if not self.opt.no_cuda:
            real_samples = real_samples.cuda()
            real_samples_cond = real_samples_cond.cuda()

        # Set input/output variables
        self.input.resize_as_(real_samples.data).copy_(real_samples.data)
        self.input_cond.resize_as_(real_samples_cond.data).copy_(real_samples_cond.data)
        self.label.resize_(self.batch_size).fill_(self.real_label)
        self.inputv = Variable(self.input)
        self.inputv_cond = Variable(self.input_cond)
        self.labelv = Variable(self.label)

    def generate_noise_vector(self, ):
        """Generate a noise vector."""
        self.noise.resize_(
            self.batch_size, int(self.opt.nz), 1, 1).normal_(0, 1)
        self.noisev = Variable(self.noise)  # TODO: Add volatile=True???

    def render_batch(self, batch, batch_cond):
        """Render a batch of splats."""
        batch_size = batch.size()[0]

        # Generate camera positions on a sphere



        # print ('Gen Min', F.tanh(batch[:, :, :1]).min().cpu().data.numpy(),
        #        'Max', F.tanh(batch[:, :, :1]).max().cpu().data.numpy(),
        #        'Mean', F.tanh(batch[:, :, :1]).mean().cpu().data.numpy())
        rendered_data = []
        rendered_data_cond = []
        rendered_res_world = []
        scenes=[]
        # Set splats into rendering scene
        if 'sphere' in self.scene['objects']:
            del self.scene['objects']['sphere']
        if 'triangle' in self.scene['objects']:
            del self.scene['objects']['triangle']

        if 'disk' not in self.scene['objects']:
            self.scene['objects'] = {'disk': {'pos': None, 'normal': None, 'material_idx': None}}
        if self.opt.fix_splat_pos:
            x, y = np.meshgrid(
                np.linspace(-1, 1, self.opt.splats_img_size),
                np.linspace(-1, 1, self.opt.splats_img_size))
        lookat = self.opt.at if self.opt.at is not None else [0.0, 0.0, 0.0, 1.0]
        self.scene['camera']['at'] = tch_var_f(lookat)
        self.scene['objects']['disk']['material_idx'] = tch_var_l(np.zeros(self.opt.splats_img_size * self.opt.splats_img_size))
        for idx in range(batch_size):
            # Get splats positions and normals
            if not self.opt.fix_splat_pos:
                pos = batch[idx][:, :3]
                pos = ((pos - torch.mean(pos, 0)) /
                       (torch.max(pos) - torch.min(pos)))
                normals = batch[idx][:, 3:]
            else:
                pos = np.stack((x.ravel(), y.ravel()), axis=1)
                pos = tch_var_f(pos)
                # TODO: Thanh here?
                #pos = torch.cat([pos, F.tanh(batch[idx][:, :1])], 1) # for NDC
                #pos = torch.cat([pos, -torch.abs(batch[idx][:, :1])], 1)  # for along-ray
                #pos = torch.cat([pos, -F.relu(batch[idx][:, :1])], 1)  # for along-ray but not explicitly < -f (can it learn to be < -f?)
                pos = torch.cat([pos, -self.scene['camera']['focal_length']-F.relu(batch[idx][:, :1])], 1)  # for along-ray
                if self.opt.norm_sph_coord:
                    # TODO: Sigmoid here?
                    #phi_theta = F.sigmoid(batch[idx][:, 1:]) * tch_var_f([2 * np.pi, np.pi / 2.])[np.newaxis, :]
                    phi = F.sigmoid(batch[idx][:, 1]) * 2 * np.pi
                    theta = F.tanh(batch[idx][:, 2]) * np.pi / 2
                    normals = sph2cart_unit(torch.stack((phi, theta), dim=1))
                else:
                    normals = batch[idx][:, 1:]

            self.scene['objects']['disk']['pos'] = pos
            self.scene['objects']['disk']['normal'] = normals

            # Set camera position
            if not self.opt.same_view:
                self.scene['camera']['eye'] = batch_cond[idx]
            else:
                self.scene['camera']['eye'] = batch_cond[0]

            # Render scene
            #res = render_splats_NDC(self.scene)
            res = render_splats_along_ray(self.scene)
            #import ipdb; ipdb.set_trace()
            res_world = cam_to_world(pos=res['pos'], normal=res['normal'], camera=self.scene['camera'])
            # print("out of cam-to-world")
            # Get rendered output
            if self.opt.render_img_nc == 1:
                depth = res['depth']
                # Normalize depth image
                cond = depth >= self.scene['camera']['far']
                depth = where(cond, torch.min(depth), depth)
                im = ((depth - torch.min(depth)) /
                      (torch.max(depth) - torch.min(depth)))
                im = im.unsqueeze(0)
            else:
                im = res['image'].permute(2, 0, 1)

            # Store normalized depth into the data
            rendered_data.append(im)
            rendered_data_cond.append(self.scene['camera']['eye'])
            rendered_res_world.append(res_world)
            scenes.append(self.scene)
        print("out of forloop")
        rendered_data = torch.stack(rendered_data)
        #rendered_res_world = torch.stack(rendered_res_world)
        #scenes=torch.stack(scenes)
        # out_file = os.path.join("/data/lisa/data/sai","generator_output"+".npy")
        # np.save(out_file,batch)
        print("saved batch")
        out_file2 = os.path.join("/data/lisa/data/sai","scene_output"+".npy")
        np.save(out_file2,scenes)
        out_file3 = os.path.join("/data/lisa/data/sai","res_world"+".npy")
        np.save(out_file3,rendered_res_world)
        #rendered_data_cond = torch.stack(torch.cat([large_scene['camera']['eye'],view_dir]))
        return rendered_data

    def sample(self, ):
        """Train networtk."""
        # Start training
        self.netG.load_state_dict(torch.load(
        open(self.opt.gen_model_path, 'rb'
        )
        ))
        iteration=0
        self.netD.load_state_dict(torch.load(
        open(self.opt.dis_model_path, 'rb'
         )
         ))


        from diffrend.numpy.ops import sph2cart_vec as np_sph2cart
        #fake = self.netG(self.noisev,self.inputv_cond)
        #fake_rendered = self.render_batch(fake,self.inputv_cond)
        phi = np.linspace(np.deg2rad(5), np.deg2rad(90), 100)
        theta = np.ones_like(phi) * np.deg2rad(45)
        cam_dist_vec = np.ones_like(phi) * self.opt.cam_dist
        cam_pos = np_sph2cart(np.stack((cam_dist_vec, phi, theta), axis=1))
        #cam_pos = np.split(cam_pos, 100 / 4)
        #for sub_batch in cam_pos:
        noise = torch.FloatTensor(int(self.opt.nz), 1, 1).cuda()
        noise.resize_( int(self.opt.nz), 1, 1).normal_(0, 1)
        #import ipdb;ipdb.set_trace()
        noise=noise.repeat(100,1,1,1)
        #import ipdb;ipdb.set_trace()
        noise = Variable(noise)
        fake = self.netG(noise,tch_var_f(cam_pos))
        fake_rendered = self.render_batch(fake,tch_var_f(cam_pos))
        torchvision.utils.save_image(fake_rendered.data, os.path.join(self.opt.out_dir,  'smooth_%d.png' % (iteration)), nrow=10, normalize=True, scale_each=True)
        for i in range(100):
            torchvision.utils.save_image(fake_rendered[i].data, os.path.join(self.opt.out_dir,  'smooth_ind_%d_%d.png' % (iteration,i)), nrow=1, normalize=True, scale_each=True)


def main():
    """Start training."""
    # Parse args
    opt = Parameters().parse()
    exp_dir = os.path.join(opt.out_dir, opt.name)
    mkdirs(exp_dir)
    exp = None
    copy_scripts_to_folder(exp_dir)
    if HYPERDASH_SUPPORTED:
        # create new Hyperdash logger
        exp = Experiment("inverse graphics")

        # log all the parameters for this experiment
        for key, val in opt.__dict__.items():
            exp.param(key, val)

    # Create dataset loader
    dataset_load = None #Dataset_load(opt)

    # Create GAN
    gan = GAN(opt, dataset_load, exp,exp_dir)

    # Train gan
    gan.sample()

    # Finsih Hyperdash logger
    if exp is not None:
        exp.end()


if __name__ == '__main__':
    main()
