"""Generator."""
from __future__ import absolute_import

import copy
import numpy as np
from scipy.misc import imsave
import os
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from imageio import imsave
import shutil
import torchvision
from diffrend.torch.GAN.datasets import Dataset_load
from diffrend.torch.GAN.twin_networks import create_networks
from diffrend.torch.GAN.parameters_halfbox_shapenet import Parameters
from diffrend.torch.GAN.utils import make_dot
from diffrend.torch.params import SCENE_BASIC, SCENE_SPHERE_HALFBOX
from diffrend.torch.utils import tch_var_f, tch_var_l, where, get_data, normalize, cam_to_world, spatial_3x3, grad_spatial2d
from diffrend.torch.renderer import render, render_splats_NDC, render_splats_along_ray
from diffrend.utils.sample_generator import uniform_sample_sphere
from diffrend.torch.ops import sph2cart_unit
from diffrend.utils.utils import contrast_stretch_percentile

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from diffrend.torch.utils import get_normalmap_image
# try: # temporarily
#     from hyperdash import Experiment
#     HYPERDASH_SUPPORTED = True
# except ImportError:
HYPERDASH_SUPPORTED = False

def copy_scripts_to_folder(expr_dir):
    shutil.copy("twin_networks.py", expr_dir)
    shutil.copy("../params.py", expr_dir)
    shutil.copy("../renderer.py", expr_dir)
    shutil.copy("objects_folder_multi.py", expr_dir)
    shutil.copy("parameters_halfbox_shapenet.py", expr_dir)
    shutil.copy(__file__, expr_dir)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def plot(data,epoch,target_dir):
    pos0 = get_data(data[0]['pos'])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos0[:, 0], pos0[:, 1], pos0[:, 2], s=1.3)
    fig.savefig(os.path.join(target_dir, '{}_plotr_1.{}'.format(epoch, 'png')))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx in range(0, len(data)):
        pos = get_data(data[idx]['pos'])
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=1.3)
    fig.savefig(os.path.join(target_dir, '{}_plotr_2.{}'.format(epoch, 'png')))
def plot2(data,epoch,target_dir):
    pos0 = get_data(data[0]['pos'])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos0[:, 0], pos0[:, 1], pos0[:, 2], s=1.3)
    fig.savefig(os.path.join(target_dir, '{}_plotr2_1.{}'.format(epoch, 'png')))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx in range(0, len(data)):
        pos = get_data(data[idx]['pos'])
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=1.3)
    fig.savefig(os.path.join(target_dir, '{}_plotr2_2.{}'.format(epoch, 'png')))
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


def calc_gradient_penalty(discriminator, real_data, fake_data, fake_data_cond, gp_lambda):
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
        if not self.opt.toy_example:
            self.dataset_load.initialize_dataset()
            self.dataset = self.dataset_load.get_dataset()
            self.dataset_load.initialize_dataset_loader(1)  # TODO: This is a hack!!!!
            self.dataset_loader = self.dataset_load.get_dataset_loader()

    def create_networks(self, ):
        """Create networks."""
        self.netG, self.netG2, self.netD, self.netD2 = create_networks(self.opt, verbose=True)
        if not self.opt.no_cuda:
            self.netD = self.netD.cuda()
            self.netG = self.netG.cuda()
            self.netD2 = self.netD2.cuda()
            self.netG2 = self.netG2.cuda()

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
        self.input2 = torch.FloatTensor(
            self.opt.batchSize, self.opt.render_img_nc,
            self.opt.render_img_size, self.opt.render_img_size)
        self.input_depth = torch.FloatTensor(
            self.opt.batchSize, 1,
            self.opt.render_img_size, self.opt.render_img_size)
        self.input_normal = torch.FloatTensor(
            self.opt.batchSize, 1,
            self.opt.render_img_size, self.opt.render_img_size)
        self.input_cond = torch.FloatTensor(
            self.opt.batchSize, 3)
        self.input_cond2 = torch.FloatTensor(
            self.opt.batchSize, 3)
        self.noise = torch.FloatTensor(
            self.opt.batchSize, int(self.opt.nz), 1, 1)
        self.noise1 = torch.FloatTensor(
            self.opt.batchSize, int(self.opt.nz), 1, 1)
        self.noise2 = torch.FloatTensor(
            self.opt.batchSize, int(self.opt.nz), 1, 1)
        self.noise21 = torch.FloatTensor(
            self.opt.batchSize, int(self.opt.nz), 1, 1)
        self.fixed_noise = torch.FloatTensor(
            self.opt.batchSize, int(self.opt.nz), 1, 1).normal_(0, 1)
        self.fixed_noise2 = torch.FloatTensor(
            self.opt.batchSize, int(self.opt.nz), 1, 1).normal_(0, 1)
        self.label = torch.FloatTensor(2*self.opt.batchSize)
        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1

        if not self.opt.no_cuda:
            self.input = self.input.cuda()
            self.input2 = self.input2.cuda()
            self.input_depth=self.input_depth.cuda()
            self.input_normal=self.input_normal.cuda()
            self.input_cond = self.input_cond.cuda()
            self.input_cond2 = self.input_cond2.cuda()
            self.label = self.label.cuda()
            self.noise = self.noise.cuda()
            self.noise1 = self.noise1.cuda()
            self.fixed_noise = self.fixed_noise.cuda()
            self.noise2 = self.noise2.cuda()
            self.noise21 = self.noise21.cuda()
            self.fixed_noise2 = self.fixed_noise2.cuda()
            self.one = self.one.cuda()
            self.mone = self.mone.cuda()

        self.fixed_noise = Variable(self.fixed_noise)
        self.fixed_noise2 = Variable(self.fixed_noise2)

    def create_criterion(self, ):
        """Create criterion."""
        self.criterion = nn.BCELoss()
        if not self.opt.no_cuda:
            self.criterion = self.criterion.cuda()

    def create_optimizers(self, ):
        """Create optimizers."""
        if self.opt.optimizer == 'adam':
            self.optimizerD = optim.Adam(self.netD.parameters(),
                                         lr=self.opt.lr,
                                         betas=(self.opt.beta1, 0.999))
            self.optimizerD2 = optim.Adam(self.netD2.parameters(),
                                         lr=self.opt.lr,
                                         betas=(self.opt.beta1, 0.999))
            self.optimizerG = optim.Adam(self.netG.parameters(),
                                         lr=self.opt.lr*0.5,
                                         betas=(self.opt.beta1, 0.999))
            self.optimizerG2 = optim.Adam(self.netG2.parameters(),
                                         lr=self.opt.lr,
                                         betas=(self.opt.beta1, 0.999))
        elif self.opt.optimizer == 'rmsprop':
            self.optimizerD = optim.RMSprop(self.netD.parameters(),
                                            lr=self.opt.lr)
            self.optimizerD2 = optim.RMSprop(self.netD2.parameters(),
                                            lr=self.opt.lr)
            self.optimizerG = optim.RMSprop(self.netG.parameters(),
                                            lr=self.opt.lr*0.5)
            self.optimizerG2 = optim.RMSprop(self.netG2.parameters(),
                                            lr=self.opt.lr)
        else:
            raise ValueError('Unknown optimizer: ' + self.opt.optimizer)
        LR_fn = None
        if self.opt.lr_sched_type == 'step':
            LR_fn = optim.lr_scheduler.StepLR
        elif self.opt.lr_sched_type == 'exp':
            LR_fn = optim.lr_scheduler.ExponentialLR

        self.optG_z_lr_scheduler = LR_fn(self.optimizerG,
                                       step_size=self.opt.z_lr_sched_step,
                                       gamma=self.opt.z_lr_sched_gamma)
        self.optG2_normal_lr_scheduler = LR_fn(self.optimizerG2,
                                        step_size=self.opt.normal_lr_sched_step,
                                        gamma=self.opt.normal_lr_sched_gamma)

        self.LR_SCHED_MAP = [self.optG_z_lr_scheduler, self.optG2_normal_lr_scheduler]
        self.OPT_MAP = [self.optimizerG, self.optimizerG2]

    def get_samples(self):
        """Get samples."""
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

        return samples

    def get_real_samples(self):
        """Get a real sample."""
        # samples = self.get_samples()

        # Define the camera poses
        if not self.opt.same_view:
            self.cam_pos = uniform_sample_sphere(radius=self.opt.cam_dist, num_samples=self.opt.batchSize,
                                                 axis=self.opt.axis, angle=np.deg2rad(self.opt.angle),
                                                 theta_range=self.opt.theta, phi_range=self.opt.phi)
            self.cam_pos2 = uniform_sample_sphere(radius=self.opt.cam_dist, num_samples=self.opt.batchSize,  #np.deg2rad(
                                                 axis=self.opt.axis, angle=np.deg2rad(self.opt.angle),
                                                 theta_range=self.opt.theta, phi_range=self.opt.phi)

        # Create a splats rendering scene
        large_scene = create_scene(self.opt.width, self.opt.height,
                                   self.opt.fovy, self.opt.focal_length,
                                   self.opt.n_splats)
        lookat = self.opt.at if self.opt.at is not None else [0.0, 0.0, 0.0, 1.0]
        large_scene['camera']['at'] = tch_var_f(lookat)

        # Render scenes
        data = []
        data_depth = []
        data_normal = []
        data_cond=[]
        inpath=self.opt.out_dir+'/'
        for idx in range(self.opt.batchSize):
            # Save the splats into the rendering scene
            if not self.opt.toy_example:
                if self.opt.use_mesh:
                    if 'sphere' in large_scene['objects']:
                        del large_scene['objects']['sphere']
                    if 'disk' in large_scene['objects']:
                        del large_scene['objects']['disk']
                    if 'triangle' not in large_scene['objects']:
                        large_scene['objects'] = {
                            'triangle': {'face': None, 'normal': None,
                                         'material_idx': None}}

                    samples = self.get_samples()

                    # # TODO: Solve this hack!!!!!!
                    # while True:
                    #     samples = self.get_samples()
                    #     if samples['mesh']['face'][0].size(0) <= 3000:
                    #         break
                    # print (samples['mesh']['face'][0].size())
                    large_scene['objects']['triangle']['material_idx'] = tch_var_l(
                        np.zeros(samples['mesh']['face'][0].shape[0], dtype=int).tolist())
                    large_scene['objects']['triangle']['face'] = Variable(
                        samples['mesh']['face'][0].cuda(), requires_grad=False)
                    large_scene['objects']['triangle']['normal'] = Variable(
                        samples['mesh']['normal'][0].cuda(),
                        requires_grad=False)
                else:
                    if 'sphere' in large_scene['objects']:
                        del large_scene['objects']['sphere']
                    if 'triangle' in large_scene['objects']:
                        del large_scene['objects']['triangle']
                    if 'disk' not in large_scene['objects']:
                        large_scene['objects'] = {'disk': {'pos': None,
                                                           'normal': None,
                                                           'material_idx': None}}
                    large_scene['objects']['disk']['radius'] = tch_var_f(
                        np.ones(self.opt.n_splats) * self.opt.splats_radius)
                    large_scene['objects']['disk']['material_idx'] = tch_var_l(
                        np.zeros(self.opt.n_splats, dtype=int).tolist())
                    large_scene['objects']['disk']['pos'] = Variable(
                        samples['splats']['pos'][idx].cuda(), requires_grad=False)
                    large_scene['objects']['disk']['normal'] = Variable(
                        samples['splats']['normal'][idx].cuda(),
                        requires_grad=False)

            # Set camera position
            if not self.opt.same_view:
                large_scene['camera']['eye'] = tch_var_f(self.cam_pos[idx])
            else:
                large_scene['camera']['eye'] = tch_var_f(self.cam_pos[0])

            y1 = self.cam_pos[idx].reshape(1,-1)
            b = np.ones((1, 1))
            light_posi = np.concatenate((y1,b), axis = 1)

            #print(light_posi.shape)
            large_scene['lights']['pos'] = tch_var_f(light_posi)

            # Render scene
            view_dir = normalize(large_scene['camera']['at'] - large_scene['camera']['eye'])
            res = render(large_scene,
                         norm_depth_image_only=self.opt.norm_depth_image_only, double_sided=True,use_quartic=self.opt.use_quartic)

            # Get rendered output
            if self.opt.render_img_nc == 1:
                depth = res['depth']
                #Normalize depth image
                # cond = depth >= large_scene['camera']['far']
                # depth = where(cond, torch.min(depth), depth)
                # im_d = ((depth - torch.min(depth)) /
                #       (torch.max(depth) - torch.min(depth)))
                im_d = depth.unsqueeze(0)
            else:
                depth = res['depth']
                #Normalize depth image
                cond = depth >= large_scene['camera']['far']
                depth = where(cond, torch.min(depth), depth)
                # im_d = ((depth - torch.min(depth)) /
                #       (torch.max(depth) - torch.min(depth)))
                im_d = depth.unsqueeze(0)
                im = res['image'].permute(2, 0, 1)

                target_normal_ = get_data(res['normal'])
                target_normalmap_img_ = get_normalmap_image(target_normal_)
                im_n=tch_var_f(target_normalmap_img_).permute(2, 0, 1)


            # Add depth image to the output structure
            if self.iterationa_no % self.opt.save_image_interval == 0:
                imsave(inpath+ str(self.iterationa_no)+ 'real_normalmap_{:05d}.png'.format(idx), target_normalmap_img_)
                #imsave(inpath+ str(self.iterationa_no)+ 'world_normalmap_{:05d}.png'.format(idx), target_worldnormalmap_img_)
            data.append(im)
            data_depth.append(im_d)
            data_normal.append(im_n)
            data_cond.append(large_scene['camera']['eye'])
        # Stack real samples
        real_samples = torch.stack(data)
        real_samples_depth = torch.stack(data_depth)
        real_samples_normal = torch.stack(data_normal)
        real_samples_cond = torch.stack(data_cond)
        self.batch_size = real_samples.size(0)
        if not self.opt.no_cuda:
            real_samples = real_samples.cuda()
            real_samples_depth = real_samples_depth.cuda()
            real_samples_normal = real_samples_normal.cuda()
            real_samples_cond = real_samples_cond.cuda()

        data2 = []
        data_depth2 = []
        data_cond2=[]
        for idx in range(self.opt.batchSize):
            # Save the splats into the rendering scene
            if not self.opt.toy_example:
                if self.opt.use_mesh:
                    if 'sphere' in large_scene['objects']:
                        del large_scene['objects']['sphere']
                    if 'disk' in large_scene['objects']:
                        del large_scene['objects']['disk']
                    if 'triangle' not in large_scene['objects']:
                        large_scene['objects'] = {
                            'triangle': {'face': None, 'normal': None,
                                         'material_idx': None}}
                    samples = self.get_samples()
                    # # TODO: Solve this hack!!!!!!
                    # while True:
                    #     samples = self.get_samples()
                    #     if samples['mesh']['face'][0].size(0) <= 3000:
                    #         break
                    # print (samples['mesh']['face'][0].size())
                    large_scene['objects']['triangle']['material_idx'] = tch_var_l(
                        np.zeros(samples['mesh']['face'][0].shape[0], dtype=int).tolist())
                    large_scene['objects']['triangle']['face'] = Variable(
                        samples['mesh']['face'][0].cuda(), requires_grad=False)
                    large_scene['objects']['triangle']['normal'] = Variable(
                        samples['mesh']['normal'][0].cuda(),
                        requires_grad=False)
                else:
                    if 'sphere' in large_scene['objects']:
                        del large_scene['objects']['sphere']
                    if 'triangle' in large_scene['objects']:
                        del large_scene['objects']['triangle']
                    if 'disk' not in large_scene['objects']:
                        large_scene['objects'] = {'disk': {'pos': None,
                                                           'normal': None,
                                                           'material_idx': None}}
                    large_scene['objects']['disk']['radius'] = tch_var_f(
                        np.ones(self.opt.n_splats) * self.opt.splats_radius)
                    large_scene['objects']['disk']['material_idx'] = tch_var_l(
                        np.zeros(self.opt.n_splats, dtype=int).tolist())
                    large_scene['objects']['disk']['pos'] = Variable(
                        samples['splats']['pos'][idx].cuda(), requires_grad=False)
                    large_scene['objects']['disk']['normal'] = Variable(
                        samples['splats']['normal'][idx].cuda(),
                        requires_grad=False)

            # Set camera position
            if not self.opt.same_view:
                large_scene['camera']['eye'] = tch_var_f(self.cam_pos2[idx])
            else:
                large_scene['camera']['eye'] = tch_var_f(self.cam_pos2[0])

            y1 = self.cam_pos2[idx].reshape(1,-1)
            b = np.ones((1, 1))
            light_posi = np.concatenate((y1,b), axis = 1)

            #print(light_posi.shape)
            large_scene['lights']['pos'] = tch_var_f(light_posi)

            # Render scene
            #view_dir = normalize(large_scene['camera']['at'] - large_scene['camera']['eye'])
            res = render(large_scene,
                         norm_depth_image_only=self.opt.norm_depth_image_only,double_sided=True,use_quartic=self.opt.use_quartic)

            # Get rendered output
            if self.opt.render_img_nc == 1:
                depth = res['depth']
                #Normalize depth image
                # cond = depth >= large_scene['camera']['far']
                # depth = where(cond, torch.min(depth), depth)
                # im_d = ((depth - torch.min(depth)) /
                #       (torch.max(depth) - torch.min(depth)))
                im_d = depth.unsqueeze(0)
            else:
                depth = res['depth']
                #Normalize depth image
                cond = depth >= large_scene['camera']['far']
                depth = where(cond, torch.min(depth), depth)
                # im_d = ((depth - torch.min(depth)) /
                #       (torch.max(depth) - torch.min(depth)))
                im_d = depth.unsqueeze(0)
                im = res['image'].permute(2, 0, 1)

            # Add depth image to the output structure
            data2.append(im)
            data_depth2.append(im_d)
            data_cond2.append(large_scene['camera']['eye'])
        # Stack real samples
        real_samples2 = torch.stack(data2)
        real_samples_depth2 = torch.stack(data_depth2)
        real_samples_cond2 = torch.stack(data_cond2)
        self.batch_size = real_samples.size(0)
        if not self.opt.no_cuda:
            real_samples2 = real_samples2.cuda()
            real_samples_depth2 = real_samples_depth2.cuda()
            real_samples_cond2 = real_samples_cond2.cuda()

        # Set input/output variables
        self.input.resize_as_(real_samples.data).copy_(real_samples.data)
        self.input2.resize_as_(real_samples2.data).copy_(real_samples2.data)
        self.input_depth.resize_as_(real_samples_depth.data).copy_(real_samples_depth.data)
        self.input_normal.resize_as_(real_samples_normal.data).copy_(real_samples_normal.data)
        self.input_cond.resize_as_(real_samples_cond.data).copy_(real_samples_cond.data)
        self.input_cond2.resize_as_(real_samples_cond2.data).copy_(real_samples_cond2.data)
        self.label.resize_(self.batch_size).fill_(self.real_label)
        self.inputv = Variable(self.input)
        self.inputv2 = Variable(self.input2)
        self.inputv_depth = Variable(self.input_depth)
        self.inputv_normal = Variable(self.input_normal)
        self.inputv_cond = Variable(self.input_cond)
        self.inputv_cond2 = Variable(self.input_cond2)
        self.labelv = Variable(self.label)

    def generate_noise_vector(self, ):
        """Generate a noise vector."""
        self.noise.resize_(
            self.batch_size, int(self.opt.nz), 1, 1).normal_(0, 1)
        self.noisev = Variable(self.noise)  # TODO: Add volatile=True???
        self.noise1.resize_(
            self.batch_size, int(self.opt.nz), 1, 1).normal_(0, 1)
        self.noisev1 = Variable(self.noise1)  # TODO: Add volatile=True???
        self.noise2.resize_(
            self.batch_size, int(self.opt.nz), 1, 1).normal_(0, 1)
        self.noisev2 = Variable(self.noise2)  # TODO: Add volatile=True???
        self.noise21.resize_(
            self.batch_size, int(self.opt.nz), 1, 1).normal_(0, 1)
        self.noisev21 = Variable(self.noise21)  # TODO: Add volatile=True???




    def render_batch(self, batch, batch_cond=None):
        """Render a batch of splats."""
        batch_size = batch.size()[0]

        # Generate camera positions on a sphere
        if not self.opt.same_view:
            cam_pos = uniform_sample_sphere(radius=self.opt.cam_dist, num_samples=self.opt.batchSize,   #np.deg2rad(
                                                 axis=self.opt.axis, angle=np.deg2rad(self.opt.angle),
                                                 theta_range=self.opt.theta, phi_range=self.opt.phi)

        rendered_data = []
        rendered_data_depth = []
        rendered_data_normal = []
        rendered_data_world_normal = []
        rendered_data_cond = []
        rendered_res_world=[]
        scenes=[]
        inpath=self.opt.out_dir+'/'
        z_min=self.scene['camera']['focal_length'] + 3
        z_max=z_min+5
        # Set splats into rendering scene
        if 'sphere' in self.scene['objects']:
            del self.scene['objects']['sphere']
        if 'triangle' in self.scene['objects']:
            del self.scene['objects']['triangle']
        if 'disk' not in self.scene['objects']:
            self.scene['objects'] = {'disk': {'pos': None, 'normal': None,
                                              'material_idx': None}}
        if self.opt.fix_splat_pos:
            x, y = np.meshgrid(np.linspace(-1, 1, self.opt.splats_img_size),
                               np.linspace(-1, 1, self.opt.splats_img_size))
        lookat = self.opt.at if self.opt.at is not None else [0.0, 0.0, 0.0, 1.0]
        self.scene['camera']['at'] = tch_var_f(lookat)
        self.scene['objects']['disk']['material_idx'] = tch_var_l(
            np.zeros(self.opt.splats_img_size * self.opt.splats_img_size))
        loss=0.0
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
                #import ipdb; ipdb.set_trace()
                # TODO: Thanh here?

                eps=1e-3
                #z = (F.relu(batch[idx][:, :1]) + z_min)
                z = batch[idx][:, :1]
                z = (z - z.min()) / (z.max() - z.min() + eps) * (z_max - z_min) + z_min

                #loss += 0.5*(torch.mean(F.relu(z_min - torch.abs(z))**2 + F.relu(torch.abs(z) - z_max)**2))

                pos = torch.cat([pos, -z], 1)


                if self.opt.norm_sph_coord:
                    # TODO: Sigmoid here?
                    # phi_theta = F.sigmoid(batch[idx][:, 1:]) * tch_var_f([2 * np.pi, np.pi / 2.])[np.newaxis, :]
                    phi = F.sigmoid(batch[idx][:, 1]) * 2 * np.pi
                    theta = F.tanh(batch[idx][:, 2]) * np.pi / 2
                    normals = sph2cart_unit(torch.stack((phi, theta), dim=1))
                else:
                    normals = batch[idx][:, 1:]

            self.scene['objects']['disk']['pos'] = pos
            self.scene['objects']['disk']['normal'] = normals

            # Set camera position
            if batch_cond is None:

                if not self.opt.same_view:
                    self.scene['camera']['eye'] = tch_var_f(cam_pos[idx])
                else:
                    self.scene['camera']['eye'] = tch_var_f(cam_pos[0])

            else:
                if not self.opt.same_view:
                    self.scene['camera']['eye'] = batch_cond[idx]
                else:
                    self.scene['camera']['eye'] = batch_cond[0]

            y1=get_data(batch_cond[idx])
            y1 = y1.reshape(1,-1)
            b = np.ones((1, 1))
            light_posi = np.concatenate((y1,b), axis = 1)
            self.scene['lights']['pos'] = tch_var_f(light_posi)

            # Render scene
            # res = render_splats_NDC(self.scene)
            res = render_splats_along_ray(self.scene,use_old_sign=self.opt.use_old_sign,use_quartic=self.opt.use_quartic)
            world_tform = cam_to_world(res['pos'].view((-1, 3)), res['normal'].view((-1, 3)), self.scene['camera'])

            # Get rendered output
            res_pos = res['pos'].contiguous()
            res_pos_2D = res_pos.view((self.opt.splats_img_size, self.opt.splats_img_size, 3))
            spatial_loss = spatial_3x3(res_pos_2D)
            spatial_var = torch.mean(res_pos[:, 0].var() + res_pos[:, 1].var() + res_pos[:, 2].var())
            loss +=  0.01 * (1 / (spatial_var + 1e-4))+0.5 * spatial_loss
            if self.opt.render_img_nc == 1:
                depth = res['depth']
                # Normalize depth image
                # cond = depth >= self.scene['camera']['far']
                # depth = where(cond, torch.min(depth), depth)
                # im = ((depth - torch.min(depth)) /
                #       (torch.max(depth) - torch.min(depth)))
                im = depth.unsqueeze(0)
            else:
                W=128
                H=128

                depth = res['depth']
                # Normalize depth image
                cond = depth >= self.scene['camera']['far']
                depth = where(cond, torch.min(depth), depth)
                #im_d = ((depth - torch.min(depth)) /
                #       (torch.max(depth) - torch.min(depth)))
                #im_d = where(cond, torch.min(depth), depth)
                im_d = depth.unsqueeze(0)
                im = res['image'].permute(2, 0, 1)

                target_normal_ = get_data(res['normal'])
                target_normalmap_img_ = get_normalmap_image(target_normal_)
                im_n=tch_var_f(target_normalmap_img_).permute(2, 0, 1)

                target_worldnormal_ = get_data(world_tform['normal']).reshape((H, W, 3))
                target_worldnormalmap_img_ = get_normalmap_image(target_normal_)
                im_wn=tch_var_f(target_worldnormalmap_img_).permute(2, 0, 1)
            if self.iterationa_no % self.opt.save_image_interval == 0:
                imsave(inpath+ str(self.iterationa_no)+ 'normalmap_{:05d}.png'.format(idx), target_normalmap_img_)
                imsave(inpath+ str(self.iterationa_no)+ 'world_normalmap_{:05d}.png'.format(idx), target_worldnormalmap_img_)
            if self.iterationa_no % 200 == 0:
                im2 = get_data(res['image'])
                depth2 = get_data(res['depth'])

                # plt.figure()
                # plt.imshow(im)

                #

                pos = get_data(res['pos'])

                out_file2 = ("pos"+".npy")
                np.save(inpath+out_file2,pos)

                out_file2 = ("im"+".npy")
                np.save(inpath+out_file2,im2)

                out_file2 = ("depth"+".npy")
                np.save(inpath+out_file2,depth2)

                #with open(inpath+ str(self.iterationa_no)+ 'normalmap_{:05d}.png'.format(idx), 'w') as fid2:


                # pos_normal = res['pos']
                # pos_normal = get_data(pos_normal)
                # filename_prefix="input"
                # with open(inpath+ '_{:05d}.xyz'.format(idx), 'w') as fid:
                #     for sub_idx in range(pos_normal.shape[0]):
                #         fid.write('{}\n'.format(' '.join([str(x) for x in pos_normal[sub_idx]])))


                pos_normal = torch.cat([res['pos'],res['normal'].view(-1, 3)],1)
                pos_normal = get_data(pos_normal)
                filename_prefix="input"
                with open(inpath+ str(self.iterationa_no)+ 'withnormal_{:05d}.xyz'.format(idx), 'w') as fid2:
                    for sub_idx in range(pos_normal.shape[0]):
                        fid2.write('{}\n'.format(' '.join([str(x) for x in pos_normal[sub_idx]])))

                pos_normal = torch.cat([world_tform['pos'],world_tform['normal']],1)
                pos_normal = get_data(pos_normal)
                filename_prefix="input"
                with open(inpath+ str(self.iterationa_no)+ 'withnormal_world_{:05d}.xyz'.format(idx), 'w') as fid2:
                    for sub_idx in range(pos_normal.shape[0]):
                        fid2.write('{}\n'.format(' '.join([str(x) for x in pos_normal[sub_idx]])))
            if self.opt.gz_gi_loss is not None and self.opt.gz_gi_loss > 0:
                gradZ = grad_spatial2d(res_pos_2D[:, :, 2][:, :, np.newaxis])
                gradImg = grad_spatial2d(torch.mean(im, dim=0)[:, :, np.newaxis])
                for (gZ, gI) in zip(gradZ, gradImg):
                    loss += self.opt.gz_gi_loss * torch.mean(torch.abs(torch.abs(gZ) - torch.abs(gI)))
            # Store normalized depth into the data
            rendered_data.append(im)
            rendered_data_depth.append(im_d)
            rendered_data_normal.append(im_n)
            rendered_data_world_normal.append(im_wn)
            #rendered_res_world.append(dict_res_world)
            rendered_data_cond.append(self.scene['camera']['eye'])
            scenes.append(self.scene)
            #scenes.append(self.scene)

        rendered_data = torch.stack(rendered_data)
        rendered_data_depth = torch.stack(rendered_data_depth)
        rendered_data_normal = torch.stack(rendered_data_normal)
        rendered_data_world_normal = torch.stack(rendered_data_world_normal)

        return rendered_data, rendered_data_depth, rendered_data_normal, rendered_data_world_normal, rendered_res_world, loss/self.opt.batchSize

    def train(self, ):
        """Train networtk."""
        # Start training
        if self.opt.gen_model_path is not None:
            print("reloading networks from")
            print(self.opt.gen_model_path)
            self.netG.load_state_dict(torch.load(
            open(self.opt.gen_model_path, 'rb'
            )
            ))
            #iteration=0
            self.netD.load_state_dict(torch.load(
            open(self.opt.dis_model_path, 'rb'
            )
            ))
        file_name = os.path.join(self.opt.out_dir, 'L2.txt')
        with open(file_name, 'wt') as l2_file:
            curr_generator_idx = 0
            for iteration in range(self.opt.n_iter):
                self.iterationa_no=iteration

                # Train Discriminator critic_iters times
                for j in range(self.opt.critic_iters):
                    # Train with real
                    #################
                    self.netD.zero_grad()
                    self.netD2.zero_grad()
                    self.get_real_samples()
                    #input_D = torch.cat([self.inputv,self.inputv_depth],1)
                    real_output = self.netD(self.inputv, self.inputv_cond)
                    real_output_depth = self.netD2(self.inputv2, self.inputv_cond2)
                    if self.opt.criterion == 'GAN':
                        errD_real = self.criterion(real_output, self.labelv)
                        errD_real.backward()
                    elif self.opt.criterion == 'WGAN':
                        errD_real = real_output.mean()
                        errD_real_depth = real_output_depth.mean()
                        errD_real.backward(self.mone)
                        errD_real_depth.backward(self.mone)
                    else:
                        raise ValueError('Unknown GAN criterium')

                    # Train with fake
                    #################
                    self.generate_noise_vector()
                    fake_z = self.netG(self.noisev, self.inputv_cond)
                    fake_n = self.netG2(self.noisev1, self.inputv_cond)
                    fake2_z = self.netG(self.noisev2, self.inputv_cond2)
                    fake2_n = self.netG2(self.noisev21, self.inputv_cond2)
                    fake=torch.cat([fake_z,fake_n],2)
                    fake2=torch.cat([fake2_z,fake2_n],2)
                    fake_rendered,fd,fn,fwn,r,loss = self.render_batch(fake, self.inputv_cond)
                    fake_rendered2,fd2,fn2,fwn2,r2,loss2 = self.render_batch(fake2, self.inputv_cond2)
                    fake_D=torch.cat([fake_rendered.detach(),fd.detach()],1)
                    # Do not bp through gen
                    outD_fake = self.netD(fake_rendered.detach(), self.inputv_cond.detach())
                    outD_fake_depth = self.netD2(fake_rendered2.detach(), self.inputv_cond2.detach())
                    if self.opt.criterion == 'GAN':
                        labelv = Variable(self.label.fill_(self.fake_label))
                        errD_fake = self.criterion(outD_fake, labelv)
                        errD_fake.backward()
                        errD = errD_real + errD_fake
                    elif self.opt.criterion == 'WGAN':
                        errD_fake = outD_fake.mean()
                        errD_fake_depth = outD_fake_depth.mean()
                        errD_fake.backward(self.one)
                        errD_fake_depth.backward(self.one)
                        errD = errD_fake - errD_real
                        errD_depth = errD_fake_depth - errD_real_depth
                    else:
                        raise ValueError('Unknown GAN criterium')

                    # Compute gradient penalty
                    if self.opt.gp != 'None':
                        gradient_penalty = calc_gradient_penalty(
                            self.netD, self.inputv.data, fake_rendered.data,self.inputv_cond.data,
                            self.opt.gp_lambda)
                        gradient_penalty.backward()
                        errD += gradient_penalty

                        gradient_penalty_depth = calc_gradient_penalty(
                            self.netD2, self.inputv2.data, fake_rendered2.data,self.inputv_cond2.data,
                            self.opt.gp_lambda)
                        gradient_penalty_depth.backward()
                        errD_depth += gradient_penalty_depth
                    gnorm_D = torch.nn.utils.clip_grad_norm(self.netD.parameters(), self.opt.max_gnorm)
                    gnorm_D2 = torch.nn.utils.clip_grad_norm(self.netD2.parameters(), self.opt.max_gnorm)
                    # Update weight
                    self.optimizerD.step()
                    self.optimizerD2.step()
                    # Clamp critic weigths if not GP and if WGAN
                    if self.opt.criterion == 'WGAN' and self.opt.gp == 'None':
                        for p in self.netD.parameters():
                            p.data.clamp_(-self.opt.clamp, self.opt.clamp)

                ############################
                # (2) Update G network
                ###########################
                # To avoid computation
                # for p in self.netD.parameters():
                #     p.requires_grad = False
                self.netG.zero_grad()
                self.netG2.zero_grad()
                self.generate_noise_vector()
                fake_z = self.netG(self.noisev, self.inputv_cond)
                fake_n = self.netG2(self.noisev1, self.inputv_cond)
                fake2_z = self.netG(self.noisev2, self.inputv_cond2)
                fake2_n = self.netG2(self.noisev21, self.inputv_cond2)
                fake=torch.cat([fake_z,fake_n],2)
                fake2=torch.cat([fake2_z,fake2_n],2)
                fake_rendered,fd,fn,fwn,r,loss = self.render_batch(fake, self.inputv_cond)
                fake_rendered2,fd2,fn,fwn2,r2,loss2 = self.render_batch(fake2, self.inputv_cond2)
                outG_fake = self.netD(fake_rendered, self.inputv_cond)
                outG_fake_depth = self.netD2(fake_rendered2, self.inputv_cond2)
                #dot = make_dot(fake)
                # dot.render('teeest/gen.gv', view=True)
                # quit()

                if self.opt.criterion == 'GAN':
                    # Fake labels are real for generator cost
                    labelv = Variable(self.label.fill_(self.real_label))
                    errG = self.criterion(outG_fake, labelv)
                    errG.backward()
                elif self.opt.criterion == 'WGAN':
                    errG = 0.5*outG_fake.mean() + 0.5*outG_fake_depth.mean()+0.5*(loss+loss2)
                    errG.backward(self.mone)
                else:
                    raise ValueError('Unknown GAN criterium')
                gnorm_G = torch.nn.utils.clip_grad_norm(self.netG.parameters(),
                                                        self.opt.max_gnorm)
                if self.opt.alt_opt_zn_interval is not None and iteration >= self.opt.alt_opt_zn_start:
                    # update one of the generators
                    if (iteration - self.opt.alt_opt_zn_start) % self.opt.alt_opt_zn_interval == 0:
                        # switch generator vars to optimize
                        curr_generator_idx = (1 - curr_generator_idx)
                    if iteration < self.opt.lr_iter:
                        self.LR_SCHED_MAP[curr_generator_idx].step()
                        self.OPT_MAP[curr_generator_idx].step()
                else:
                    if iteration < self.opt.lr_iter:
                        self.optG_z_lr_scheduler.step()
                        self.optG2_normal_lr_scheduler.step()
                    self.optimizerG.step()
                    self.optimizerG2.step()

                # Log print
                mse_criterion = nn.MSELoss().cuda()

                if iteration % self.opt.print_interval == 0:
                    fake_rendered_cond2,fd2,fn2,fwn2,r2,loss3 = self.render_batch(fake, self.inputv_cond)
                    l2_loss=mse_criterion(fd2, self.inputv_depth)
                    Wassertein_D = (errD_real.data[0] - errD_fake.data[0])
                    Wassertein_D_depth = (errD_real_depth.data[0] - errD_fake_depth.data[0])
                    print('\n[%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_D_real: %.4f'
                          ' Loss_D_fake: %.4f Loss_D_real_depth: %.4f Loss_D_fake_depth: %.4f Wassertein D: %.4f '
                          'Wassertein_depth D_depth: %.4f L2_loss: %.4f z_lr: %.8f, n_lr: %.8f' % (
                          iteration, self.opt.n_iter, errD.data[0],
                          errG.data[0], errD_real.data[0], errD_fake.data[0],errD_real_depth.data[0],errD_fake_depth.data[0],
                          Wassertein_D, Wassertein_D_depth, 0.5*(loss+loss2).data[0],
                          self.optG_z_lr_scheduler.get_lr()[0], self.optG2_normal_lr_scheduler.get_lr()[0]))
                    l2_file.write('%s\n' % (str(l2_loss.data[0])))
                    l2_file.flush()
                    print("written to file",str(l2_loss.data[0]))

                    if self.exp is not None:
                        self.exp.metric("iteration", iteration)
                        self.exp.metric("loss D", errD.data[0])
                        self.exp.metric("loss G", errG.data[0])
                        self.exp.metric("Loss D real", errD_real.data[0])
                        self.exp.metric("Loss D fake", errD_fake.data[0])
                        self.exp.metric("Wassertein D", Wassertein_D)

                # Save images
                if iteration % self.opt.save_image_interval == 0:
                    cs=tch_var_f(contrast_stretch_percentile(get_data(fd),  200, [fd.data.min(), fd.data.max()]))
                    torchvision.utils.save_image(self.inputv.data, os.path.join(self.opt.out_dir,  'input_%d.png' % (iteration)), nrow=2, normalize=True, scale_each=True)
                    #torchvision.utils.save_image(self.inputv_normal.data, os.path.join(self.opt.out_dir,  'input_normal%d.png' % (iteration)), nrow=2, normalize=False, scale_each=False)
                    torchvision.utils.save_image(fake_rendered.data, os.path.join(self.opt.out_dir,  'output_%d.png' % (iteration)), nrow=2, normalize=True, scale_each=True)
                    torchvision.utils.save_image(self.inputv_depth.data, os.path.join(self.opt.out_dir,  'input_depth%d.png' % (iteration)), nrow=2, normalize=True, scale_each=True)
                    torchvision.utils.save_image(fd.data, os.path.join(self.opt.out_dir,  'output_depth%d.png' % (iteration)), nrow=2, normalize=True, scale_each=True)
                    #torchvision.utils.save_image(fn.data, os.path.join(self.opt.out_dir,  'output_normal%d.png' % (iteration)), nrow=2, normalize=False, scale_each=False)
                    #torchvision.utils.save_image(fwn.data, os.path.join(self.opt.out_dir,  'output_world_normal%d.png' % (iteration)), nrow=2, normalize=False, scale_each=False)
                    torchvision.utils.save_image(cs.data, os.path.join(self.opt.out_dir,  'output_depth_cs%d.png' % (iteration)), nrow=2, normalize=True, scale_each=True)




                # Do checkpointing
                if iteration % self.opt.save_interval == 0:
                    self.save_networks(iteration)



    def save_networks(self, epoch):
        """Save networks to hard disk."""
        torch.save(self.netG.state_dict(),
                   '%s/netG_epoch_%d.pth' % (self.opt.out_dir, epoch))
        torch.save(self.netG2.state_dict(),
                   '%s/netG2_epoch_%d.pth' % (self.opt.out_dir, epoch))
        torch.save(self.netD.state_dict(),
                   '%s/netD_epoch_%d.pth' % (self.opt.out_dir, epoch))
        torch.save(self.netD2.state_dict(),
                   '%s/netD2_epoch_%d.pth' % (self.opt.out_dir, epoch))

    def save_images(self, epoch, input, output):
        """Save images."""
        if self.opt.render_img_nc == 1:
            imsave(self.opt.out_dir + '/input2' + str(epoch) + '.png',
                   np.uint8(255. * input.cpu().data.numpy().squeeze()))
            imsave(self.opt.out_dir + '/fz' + str(epoch) + '.png',
                   np.uint8(255. * output.cpu().data.numpy().squeeze()))
        else:
            imsave(self.opt.out_dir + '/input2' + str(epoch) + '.png',
                   np.uint8(255. * input.cpu().data.numpy().transpose((1, 2, 0))))
            imsave(self.opt.out_dir + '/output2' + str(epoch) + '.png',
                   np.uint8(255. * output.cpu().data.numpy().transpose((1, 2, 0))))


def main():
    """Start training."""
    # Parse args
    #opt = Parameters().parse()
    opt = Parameters().parse()
    exp_dir = os.path.join(opt.out_dir, opt.name)
    mkdirs(exp_dir)
    exp = None
    copy_scripts_to_folder(exp_dir)
    #exp = None
    file_name = os.path.join(exp_dir, 'opt.txt')
    args = vars(opt)
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
    if HYPERDASH_SUPPORTED:
        # create new Hyperdash logger
        exp = Experiment("inverse graphics")

        # log all the parameters for this experiment
        for key, val in opt.__dict__.items():
            exp.param(key, val)

    # Create dataset loader
    if opt.toy_example:
        dataset_load = None
    else:
        dataset_load = Dataset_load(opt)

    # Create GAN
    gan = GAN(opt, dataset_load, exp, exp_dir)

    # Train gan
    gan.train()

    # Finsih Hyperdash logger
    if exp is not None:
        exp.end()


if __name__ == '__main__':
    main()
