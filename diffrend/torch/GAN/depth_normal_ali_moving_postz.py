"""Generator."""
from __future__ import absolute_import

import copy
import numpy as np
# from scipy.misc import imsave
from imageio import imsave
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from diffrend.torch.GAN.datasets import Dataset_load
from diffrend.torch.GAN.twin_networks import create_networks
from diffrend.torch.GAN.parameters_halfbox_shapenet import Parameters
from diffrend.torch.params import SCENE_SPHERE_HALFBOX_0
from diffrend.torch.utils import (tch_var_f, tch_var_l, get_data,
                                  get_normalmap_image, cam_to_world,
                                  unit_norm2_L2loss, normal_consistency_cost,
                                  away_from_camera_penalty, spatial_3x3,
                                  depth_rgb_gradient_consistency,
                                  grad_spatial2d)
from diffrend.torch.renderer import (render, render_splats_along_ray,
                                     z_to_pcl_CC)
from diffrend.torch.NEstNet import NEstNetV1_2
from diffrend.utils.sample_generator import uniform_sample_sphere
from diffrend.utils.utils import contrast_stretch_percentile, save_xyz
from tensorboardX import SummaryWriter

import matplotlib
matplotlib.use('Agg')
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import axes3d

"""
 z_noise -> z_distance_generator -> proj_to_scene_in_CC ->
 normal_estimator_network -> render -> discriminator
 real_data -> discriminator
"""


def copy_scripts_to_folder(expr_dir):
    """Copy scripts."""
    shutil.copy("twin_networks.py", expr_dir)
    shutil.copy("../NEstNet.py", expr_dir)
    shutil.copy("../params.py", expr_dir)
    shutil.copy("../renderer.py", expr_dir)
    shutil.copy("objects_folder_multi.py", expr_dir)
    shutil.copy("parameters_halfbox_shapenet.py", expr_dir)
    shutil.copy(__file__, expr_dir)
def gauss_reparametrize(mu, logvar, n_sample=1):
    """Gaussian reparametrization"""
    std = logvar.mul(0.5).exp_()
    size = std.size()
    eps = Variable(std.data.new(size[0], n_sample, size[1]).normal_())
    z = eps.mul(std[:, None, :]).add_(mu[:, None, :])
    z = torch.clamp(z, -4., 4.)
    return z.view(z.size(0)*z.size(1), z.size(2), 1, 1)

def mkdirs(paths):
    """Create paths."""
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """Create a directory."""
    if not os.path.exists(path):
        os.makedirs(path)


def create_scene(width, height, fovy, focal_length, n_samples):
    """Create a semi-empty scene with camera parameters."""
    # Create a splats rendering scene
    scene = copy.deepcopy(SCENE_SPHERE_HALFBOX_0)

    # Define the camera parameters
    scene['camera']['viewport'] = [0, 0, width, height]
    scene['camera']['fovy'] = np.deg2rad(fovy)
    scene['camera']['focal_length'] = focal_length

    return scene


def calc_gradient_penalty(discriminator, real_data, fake_data, fake_data_cond,
                          gp_lambda):
    """Calculate GP."""
    assert real_data.size(0) == fake_data.size(0)
    alpha = torch.rand(real_data.size(0), 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = Variable(alpha * real_data + ((1 - alpha) * fake_data),
                            requires_grad=True)
    interpolates_cond = Variable(fake_data_cond, requires_grad=True)
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

    def __init__(self, opt, dataset_load=None, exp_dir=None):
        """Constructor."""
        # Save variables
        self.opt = opt
        self.dataset_load = dataset_load
        self.opt.out_dir = exp_dir

        # Define other variables
        self.real_label = 1
        self.fake_label = 0

        # Losses file
        file_name = os.path.join(self.opt.out_dir, 'losses.txt')
        self.output_loss_file = open(file_name, "wt")

        # TODO: Add comment
        if self.opt.full_sphere_sampling:
            self.opt.phi = None
            self.opt.theta = None
            self.opt.cam_dist = self.opt.cam_dist + 0.2
        else:
            self.opt.angle = None


        # TensorboardX
        self.writer = SummaryWriter(self.opt.vis_monitoring)
        print(self.opt.vis_monitoring)
        print(self.opt.out_dir)
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
            # self.cam_pos = uniform_sample_sphere(radius=self.opt.cam_dist,
            #                                      num_samples=1)
            arrays = [np.asarray([3., 3., 3.]) for _ in
                      range(self.opt.batchSize)]  # TODO: Magic numbers
            self.cam_pos = np.stack(arrays, axis=0)

        # Create dataset loader
        self.dataset_load.initialize_dataset()
        self.dataset = self.dataset_load.get_dataset()
        self.dataset_load.initialize_dataset_loader(1)  # TODO: Hack
        self.dataset_loader = self.dataset_load.get_dataset_loader()

    def create_networks(self, ):
        """Create networks."""
        self.netG, self.netG2, self.netD, self.netD2 = create_networks(
            self.opt, verbose=True, depth_only=True)  # TODO: Remove D2 and G2
        # Create the normal estimation network which takes pointclouds in the
        # camera space and outputs the normals
        assert self.netG2 is None
        self.sph_normals = True
        self.netG2 = NEstNetV1_2(sph=self.sph_normals)
        print(self.netG2)
        if not self.opt.no_cuda:
            self.netD = self.netD.cuda()
            self.netG = self.netG.cuda()
            #self.netE = self.netE.cuda()
            self.netG2 = self.netG2.cuda()

    def create_scene(self, ):
        """Create a semi-empty scene with camera parameters."""
        self.scene = create_scene(
            self.opt.splats_img_size, self.opt.splats_img_size, self.opt.fovy,
            self.opt.focal_length, self.opt.n_splats)

    def create_tensors(self, ):
        """Create the tensors."""
        # Create tensors
        self.input = torch.FloatTensor(
            self.opt.batchSize, self.opt.render_img_nc,
            self.opt.render_img_size, self.opt.render_img_size)
        self.input_depth = torch.FloatTensor(
            self.opt.batchSize, 1, self.opt.render_img_size,
            self.opt.render_img_size)
        self.input_normal = torch.FloatTensor(
            self.opt.batchSize, 1, self.opt.render_img_size,
            self.opt.render_img_size)
        self.input_cond = torch.FloatTensor(self.opt.batchSize, 3)

        self.noise = torch.FloatTensor(
            self.opt.batchSize, int(self.opt.nz), 1, 1)
        self.fixed_noise = torch.FloatTensor(
            self.opt.batchSize, int(self.opt.nz), 1, 1).normal_(0, 1)

        self.label = torch.FloatTensor(2*self.opt.batchSize)
        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1

        # Move them to the GPU
        if not self.opt.no_cuda:
            self.input = self.input.cuda()
            self.input_depth = self.input_depth.cuda()
            self.input_normal = self.input_normal.cuda()
            self.input_cond = self.input_cond.cuda()

            self.label = self.label.cuda()
            self.noise = self.noise.cuda()
            self.fixed_noise = self.fixed_noise.cuda()

            self.one = self.one.cuda()
            self.mone = self.mone.cuda()

        self.fixed_noise = Variable(self.fixed_noise)  # TODO: Why?

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
            self.optimizerG = optim.Adam(self.netG.parameters(),
                                         lr=self.opt.lr,
                                         betas=(self.opt.beta1, 0.999))
            self.optimizerG2 = optim.Adam(self.netG2.parameters(),
                                          lr=self.opt.lr,
                                          betas=(self.opt.beta1, 0.999))
        elif self.opt.optimizer == 'rmsprop':
            self.optimizerD = optim.RMSprop(self.netD.parameters(),
                                            lr=self.opt.lr)
            self.optimizerG = optim.RMSprop(self.netG.parameters(),
                                            lr=self.opt.lr)
            self.optimizerG2 = optim.RMSprop(self.netG2.parameters(),
                                             lr=self.opt.lr)
        else:
            raise ValueError('Unknown optimizer: ' + self.opt.optimizer)

        # Create the schedulers
        if self.opt.lr_sched_type == 'step':
            LR_fn = optim.lr_scheduler.StepLR
        elif self.opt.lr_sched_type == 'exp':
            LR_fn = optim.lr_scheduler.ExponentialLR
        elif self.opt.lr_sched_type is None:
            LR_fn = None
        else:
            raise ValueError('Unknown scheduler')

        self.optG_z_lr_scheduler = LR_fn(
            self.optimizerG, step_size=self.opt.z_lr_sched_step,
            gamma=self.opt.z_lr_sched_gamma)
        self.optG2_normal_lr_scheduler = LR_fn(
            self.optimizerG2, step_size=self.opt.normal_lr_sched_step,
            gamma=self.opt.normal_lr_sched_gamma)
        self.LR_SCHED_MAP = [self.optG_z_lr_scheduler,
                             self.optG2_normal_lr_scheduler]
        self.OPT_MAP = [self.optimizerG, self.optimizerG2]

    def get_samples(self):
        """Get samples."""
        try:
            samples = self.data_iter.next()
        except StopIteration:
            del self.data_iter
            self.data_iter = iter(self.dataset_loader)
            samples = self.data_iter.next()
        except AttributeError:
            self.data_iter = iter(self.dataset_loader)
            samples = self.data_iter.next()
        return samples

    def get_real_samples(self):
        """Get a real sample."""
        # Define the camera poses


            # TODO: deg2rad in all the angles????

        # Create a splats rendering scene
        large_scene = create_scene(self.opt.width, self.opt.height,
                                   self.opt.fovy, self.opt.focal_length,
                                   self.opt.n_splats)
        lookat = self.opt.at if self.opt.at is not None else [0.0, 0.0, 0.0, 1.0]
        large_scene['camera']['at'] = tch_var_f(lookat)

        # Render scenes
        data, data_depth, data_normal, data_cond = [], [], [], []
        inpath = self.opt.vis_images + '/'
        for idx in range(self.opt.batchSize):
            # Save the splats into the rendering scene
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

                large_scene['objects']['triangle']['material_idx'] = tch_var_l(
                    np.zeros(samples['mesh']['face'][0].shape[0],
                             dtype=int).tolist())
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
                    large_scene['objects'] = {
                        'disk': {'pos': None,
                                 'normal': None,
                                 'material_idx': None}}
                large_scene['objects']['disk']['radius'] = tch_var_f(
                    np.ones(self.opt.n_splats) * self.opt.splats_radius)
                large_scene['objects']['disk']['material_idx'] = tch_var_l(
                    np.zeros(self.opt.n_splats, dtype=int).tolist())
                large_scene['objects']['disk']['pos'] = Variable(
                    samples['splats']['pos'][idx].cuda(),
                    requires_grad=False)
                large_scene['objects']['disk']['normal'] = Variable(
                    samples['splats']['normal'][idx].cuda(),
                    requires_grad=False)

            # Set camera position
            if not self.opt.same_view:
                large_scene['camera']['eye'] = tch_var_f(self.cam_pos[idx])
            else:
                large_scene['camera']['eye'] = tch_var_f(self.cam_pos[0])

            large_scene['lights']['pos'][0,:3]=tch_var_f(self.light_pos1[idx])

            # Render scene
            res = render(large_scene,
                         norm_depth_image_only=self.opt.norm_depth_image_only,
                         double_sided=True, use_quartic=self.opt.use_quartic)

            # Get rendered output
            if self.opt.render_img_nc == 1:
                depth = res['depth']
                im_d = depth.unsqueeze(0)
            else:
                depth = res['depth']
                im_d = depth.unsqueeze(0)
                im = res['image'].permute(2, 0, 1)
                target_normal_ = get_data(res['normal'])
                target_normalmap_img_ = get_normalmap_image(target_normal_)
                im_n = tch_var_f(
                    target_normalmap_img_).view(im.shape[1], im.shape[2],
                                                3).permute(2, 0, 1)

            # Add depth image to the output structure
            if self.iterationa_no % self.opt.save_image_interval == 0:
                imsave((inpath + str(self.iterationa_no) +
                        'real_normalmap_{:05d}.png'.format(idx)),
                       target_normalmap_img_)
                imsave((inpath + str(self.iterationa_no) +
                        'real_depth_{:05d}.png'.format(idx)), get_data(depth))
                # imsave(inpath + str(self.iterationa_no) + 'real_depthmap_{:05d}.png'.format(idx), im_d)
                # imsave(inpath + str(self.iterationa_no) + 'world_normalmap_{:05d}.png'.format(idx), target_worldnormalmap_img_)
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

        # Set input/output variables

        self.input.resize_as_(real_samples.data).copy_(real_samples.data)
        self.input_depth.resize_as_(real_samples_depth.data).copy_(real_samples_depth.data)
        self.input_normal.resize_as_(real_samples_normal.data).copy_(real_samples_normal.data)
        self.input_cond.resize_as_(real_samples_cond.data).copy_(real_samples_cond.data)
        self.label.resize_(self.batch_size).fill_(self.real_label)
        # TODO: Remove Variables
        self.inputv = Variable(self.input)
        self.inputv_depth = Variable(self.input_depth)
        self.inputv_normal = Variable(self.input_normal)
        self.inputv_cond = Variable(self.input_cond)
        self.labelv = Variable(self.label)

    def generate_noise_vector(self, ):
        """Generate a noise vector."""
        self.noise.resize_(
            self.opt.batchSize, int(self.opt.nz), 1, 1).normal_(0, 1)
        self.noisev = Variable(self.noise)  # TODO: Add volatile=True???

    def generate_normals(self, z_batch, cam_pos, camera):
        """Generate normals from depth."""
        W, H = camera['viewport'][2:]
        normals = []
        for z, eye in zip(z_batch, cam_pos):
            camera['eye'] = eye
            pcl = z_to_pcl_CC(z.squeeze(), camera)
            n = self.netG2(pcl.view(H, W, 3).permute(2, 0, 1)[np.newaxis, ...])
            n = n.squeeze().permute(1, 2, 0).view(-1, 3).contiguous()
            normals.append(n)
        return torch.stack(normals)

    def tensorboard_pos_hook(self, grad):

        self.writer.add_image("position_gradient_im",
                                torch.sqrt(torch.sum(grad ** 2, dim=-1)),

                               self.iterationa_no)
        self.writer.add_scalar("position_mean_channel1",
                               get_data(torch.mean(grad[:,:,0])),
                               self.iterationa_no)
        self.writer.add_scalar("position_gradient_mean_channel2",
                               get_data(torch.mean(grad[:,:,1])),
                               self.iterationa_no)
        self.writer.add_scalar("position_gradient_mean_channel3",
                               get_data(torch.mean(grad[:,:,2])),
                               self.iterationa_no)
        self.writer.add_scalar("position_gradient_mean",
                               get_data(torch.mean(grad)),
                               self.iterationa_no)
        self.writer.add_histogram("position_gradient_hist_channel1", grad[:,:,0].clone().cpu().data.numpy(),self.iterationa_no)
        self.writer.add_histogram("position_gradient_hist_channel2", grad[:,:,1].clone().cpu().data.numpy(),self.iterationa_no)
        self.writer.add_histogram("position_gradient_hist_channel3", grad[:,:,2].clone().cpu().data.numpy(),self.iterationa_no)
        self.writer.add_histogram("position_gradient_hist_norm", torch.sqrt(torch.sum(grad ** 2, dim=-1)).clone().cpu().data.numpy(),self.iterationa_no)
        #print('grad', grad)

    def tensorboard_normal_hook(self, grad):

        self.writer.add_image("normal_gradient_im",
                                torch.sqrt(torch.sum(grad ** 2, dim=-1)),

                               self.iterationa_no)
        self.writer.add_scalar("normal_gradient_mean_channel1",
                               get_data(torch.mean(grad[:,:,0])),
                               self.iterationa_no)
        self.writer.add_scalar("normal_gradient_mean_channel2",
                               get_data(torch.mean(grad[:,:,1])),
                               self.iterationa_no)
        self.writer.add_scalar("normal_gradient_mean_channel3",
                               get_data(torch.mean(grad[:,:,2])),
                               self.iterationa_no)
        self.writer.add_scalar("normal_gradient_mean",
                               get_data(torch.mean(grad)),
                               self.iterationa_no)
        self.writer.add_histogram("normal_gradient_hist_channel1", grad[:,:,0].clone().cpu().data.numpy(),self.iterationa_no)
        self.writer.add_histogram("normal_gradient_hist_channel2", grad[:,:,1].clone().cpu().data.numpy(),self.iterationa_no)
        self.writer.add_histogram("normal_gradient_hist_channel3", grad[:,:,2].clone().cpu().data.numpy(),self.iterationa_no)
        self.writer.add_histogram("normal_gradient_hist_norm", torch.sqrt(torch.sum(grad ** 2, dim=-1)).clone().cpu().data.numpy(),self.iterationa_no)
        #print('grad', grad)

    def tensorboard_z_hook(self, grad):

        self.writer.add_scalar("z_gradient_mean",
                               get_data(torch.mean(grad)),
                               self.iterationa_no)
        self.writer.add_histogram("z_gradient_hist_channel", grad.clone().cpu().data.numpy(),self.iterationa_no)

        self.writer.add_image("z_gradient_im",
                               grad,
                               self.iterationa_no)


    def render_batch(self, batch, batch_cond=None):
        """Render a batch of splats."""
        batch_size = batch.size()[0]

        # Generate camera positions on a sphere



        rendered_data = []
        rendered_data_depth = []
        rendered_data_cond = []
        scenes = []
        inpath = self.opt.vis_images + '/'
        inpath_xyz = self.opt.vis_xyz + '/'
        z_min = self.scene['camera']['focal_length']
        z_max = z_min + 3

        # TODO (fmannan): Move this in init. This only needs to be done once!
        # Set splats into rendering scene
        if 'sphere' in self.scene['objects']:
            del self.scene['objects']['sphere']
        if 'triangle' in self.scene['objects']:
            del self.scene['objects']['triangle']
        if 'disk' not in self.scene['objects']:
            self.scene['objects'] = {'disk': {'pos': None, 'normal': None,
                                              'material_idx': None}}
        lookat = self.opt.at if self.opt.at is not None else [0.0, 0.0, 0.0, 1.0]
        self.scene['camera']['at'] = tch_var_f(lookat)
        self.scene['objects']['disk']['material_idx'] = tch_var_l(
            np.zeros(self.opt.splats_img_size * self.opt.splats_img_size))
        loss = 0.0
        loss_ = 0.0
        z_loss_ = 0.0
        z_norm_loss_ = 0.0
        spatial_loss_ = 0.0
        spatial_var_loss_ = 0.0
        unit_normal_loss_ = 0.0
        normal_away_from_cam_loss_ = 0.0
        image_depth_consistency_loss_ = 0.0
        for idx in range(batch_size):
            # Get splats positions and normals
            eps = 1e-3
            if self.opt.rescaled:
                z = F.relu(-batch[idx][:, 0]) + z_min
                z = ((z - z.min()) / (z.max() - z.min() + eps) *
                     (z_max - z_min) + z_min)
                pos = -z
            else:
                z = F.relu(-batch[idx][:, 0]) + z_min
                pos = -F.relu(-batch[idx][:, 0]) - z_min
            normals = batch[idx][:, 1:]

            self.scene['objects']['disk']['pos'] = pos

            # Normal estimation network and est_normals don't go together
            self.scene['objects']['disk']['normal'] = normals if self.opt.est_normals is False else None

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

            self.scene['lights']['pos'][0,:3]=tch_var_f(self.light_pos1[0])

            # Render scene
            # res = render_splats_NDC(self.scene)
            res = render_splats_along_ray(self.scene,
                                          samples=self.opt.pixel_samples,
                                          normal_estimation_method='plane')

            world_tform = cam_to_world(res['pos'].view((-1, 3)),
                                       res['normal'].view((-1, 3)),
                                       self.scene['camera'])

            # Get rendered output
            res_pos = res['pos'].contiguous()
            res_pos_2D = res_pos.view(res['image'].shape)
            # The z_loss needs to be applied after supersampling
            # TODO: Enable this (currently final loss becomes NaN!!)
            # loss += torch.mean(
            #    (10 * F.relu(z_min - torch.abs(res_pos[..., 2]))) ** 2 +
            #    (10 * F.relu(torch.abs(res_pos[..., 2]) - z_max)) ** 2)


            res_normal = res['normal']
            # depth_grad_loss = spatial_3x3(res['depth'][..., np.newaxis])
            # grad_img = grad_spatial2d(torch.mean(res['image'], dim=-1)[..., np.newaxis])
            # grad_depth_img = grad_spatial2d(res['depth'][..., np.newaxis])
            image_depth_consistency_loss = depth_rgb_gradient_consistency(
                res['image'], res['depth'])
            unit_normal_loss = unit_norm2_L2loss(res_normal, 10.0)  # TODO: MN
            normal_away_from_cam_loss = away_from_camera_penalty(
                res_pos, res_normal)
            z_pos = res_pos[..., 2]
            z_loss = torch.mean((2 * F.relu(z_min - torch.abs(z_pos))) ** 2 +
                                (2 * F.relu(torch.abs(z_pos) - z_max)) ** 2)
            z_norm_loss = normal_consistency_cost(
                res_pos, res['normal'], norm=1)
            spatial_loss = spatial_3x3(res_pos_2D)
            spatial_var = torch.mean(res_pos[..., 0].var() +
                                     res_pos[..., 1].var() +
                                     res_pos[..., 2].var())
            spatial_var_loss = (1 / (spatial_var + 1e-4))

            loss = (self.opt.zloss * z_loss +
                    self.opt.unit_normalloss*unit_normal_loss +
                    self.opt.normal_consistency_loss_weight * z_norm_loss +
                    self.opt.spatial_var_loss_weight * spatial_var_loss +
                    self.opt.grad_img_depth_loss*image_depth_consistency_loss +
                    self.opt.spatial_loss_weight * spatial_loss)
            pos_out_ = get_data(res['pos'])
            loss_ += get_data(loss)
            z_loss_ += get_data(z_loss)
            z_norm_loss_ += get_data(z_norm_loss)
            spatial_loss_ += get_data(spatial_loss)
            spatial_var_loss_ += get_data(spatial_var_loss)
            unit_normal_loss_ += get_data(unit_normal_loss)
            normal_away_from_cam_loss_ += get_data(normal_away_from_cam_loss)
            image_depth_consistency_loss_ += get_data(
                image_depth_consistency_loss)
            normals_ = get_data(res_normal)

            if self.opt.render_img_nc == 1:
                depth = res['depth']
                im = depth.unsqueeze(0)
            else:
                depth = res['depth']
                im_d = depth.unsqueeze(0)
                im = res['image'].permute(2, 0, 1)
                H, W = im.shape[1:]
                target_normal_ = get_data(res['normal']).reshape((H, W, 3))
                target_normalmap_img_ = get_normalmap_image(target_normal_)
                target_worldnormal_ = get_data(world_tform['normal']).reshape(
                    (H, W, 3))
                target_worldnormalmap_img_ = get_normalmap_image(
                    target_worldnormal_)



            imsave((inpath + str(self.iterationa_no) +
                    'depthmap_{:05d}.png'.format(idx)),
                   get_data(res['depth']))
            imsave((inpath + str(self.iterationa_no) +
                    'world_normalmap_{:05d}.png'.format(idx)),
                   target_worldnormalmap_img_)
            imsave((inpath + str(self.iterationa_no) +
                    'output_{:05d}.png'.format(idx)),
                   get_data(res['image']))
            if self.iterationa_no % 200 == 0:
                im2 = get_data(res['image'])
                depth2 = get_data(res['depth'])
                pos = get_data(res['pos'])

                out_file2 = ("pos"+".npy")
                np.save(inpath_xyz+out_file2, pos)

                out_file2 = ("im"+".npy")
                np.save(inpath_xyz+out_file2, im2)

                out_file2 = ("depth"+".npy")
                np.save(inpath_xyz+out_file2, depth2)

                # Save xyz file


                # Save xyz file in world coordinates
                save_xyz((inpath_xyz + str(self.iterationa_no) +
                          'withnormal_world_{:05d}.xyz'.format(idx)),
                         pos=get_data(world_tform['pos']),
                         normal=get_data(world_tform['normal']))
            if self.opt.gz_gi_loss is not None and self.opt.gz_gi_loss > 0:
                gradZ = grad_spatial2d(res_pos_2D[:, :, 2][:, :, np.newaxis])
                gradImg = grad_spatial2d(torch.mean(im,
                                                    dim=0)[:, :, np.newaxis])
                for (gZ, gI) in zip(gradZ, gradImg):
                    loss += (self.opt.gz_gi_loss * torch.mean(torch.abs(
                                torch.abs(gZ) - torch.abs(gI))))
            # Store normalized depth into the data
            rendered_data.append(im)
            rendered_data_depth.append(im_d)
            rendered_data_cond.append(self.scene['camera']['eye'])
            scenes.append(self.scene)

        rendered_data = torch.stack(rendered_data)
        rendered_data_depth = torch.stack(rendered_data_depth)


        return rendered_data, rendered_data_depth, loss/self.opt.batchSize
    def tensorboard_hook(self, grad):
        self.writer.add_scalar("z_gradient_mean",
                               get_data(torch.mean(grad[0])),
                               self.iterationa_no)
        self.writer.add_histogram("z_gradient_hist_channel", grad[0].clone().cpu().data.numpy(),self.iterationa_no)

        self.writer.add_image("z_gradient_im",
                               grad[0].view(128,128),
                               self.iterationa_no)

    def train(self, ):
        """Train network."""
        # Load pretrained model if required
        iteration=0
        self.iterationa_no = 0
        self.iterationa_no = 0
        if self.opt.gen_model_path is not None:
            print("Reloading networks from")
            print(' > Generator', self.opt.gen_model_path)
            self.netG.load_state_dict(
                torch.load(open(self.opt.gen_model_path, 'rb')))


            print(' > Discriminator', self.opt.dis_model_path)
            self.netD.load_state_dict(
                torch.load(open(self.opt.dis_model_path, 'rb')))

            from diffrend.numpy.ops import sph2cart_vec as np_sph2cart
        #fake = self.netG(self.noisev,self.inputv_cond)
        #fake_rendered = self.render_batch(fake,self.inputv_cond)
        if not self.opt.same_view:
            if self.opt.full_sphere_sampling:
                self.cam_pos = uniform_sample_sphere(
                    radius=self.opt.cam_dist, num_samples=self.opt.batchSize,
                    axis=None, angle=self.opt.angle,
                    theta_range=np.deg2rad(self.opt.theta), phi_range=np.deg2rad(self.opt.phi))
            else:
                self.cam_pos = uniform_sample_sphere(
                    radius=self.opt.cam_dist, num_samples=self.opt.batchSize,
                    axis=None, angle=self.opt.angle,
                    theta_range=np.deg2rad(self.opt.theta),
                    phi_range=np.deg2rad(self.opt.phi))

        self.light_pos1 = uniform_sample_sphere(
            radius=self.opt.cam_dist, num_samples=self.opt.batchSize,
            axis=None, angle=self.opt.angle,
            theta_range=np.deg2rad(self.opt.theta),
            phi_range=np.deg2rad(self.opt.phi))



        # else:
        #     z_real = mu_z.view(mu_z.size(0), mu_z.size(1), 1, 1)
        #     logvar_z = logvar_z * 0.0
        self.generate_noise_vector()
        fake_z1 = self.netG(self.noisev, tch_var_f(self.cam_pos))
        n1=self.noisev.clone()

        # The normal generator is dependent on z
        fake_n1 = self.generate_normals(fake_z1, tch_var_f(self.cam_pos),
                                       self.scene['camera'])
        fake1 = torch.cat([fake_z1, fake_n1], 2)
        fake_recon1, fd_recon, loss = self.render_batch(
            fake1, tch_var_f(self.cam_pos))
        # torchvision.utils.save_image(
        #     self.inputv.data, os.path.join(
        #         self.opt.vis_images, 'input_%d.png' % (iteration)),
        #     nrow=2, normalize=True, scale_each=True)
        torchvision.utils.save_image(fake_recon1.data, os.path.join(self.opt.out_dir,  'reconstructed_%d.png' % (iteration)), nrow=2, normalize=True, scale_each=True)

        iteration=1
        self.iterationa_no = 1
        self.iterationa_no = 1
        self.generate_noise_vector()
        print(self.noisev[0])
        print("###########################################")
        print(self.cam_pos[0])
        print("###########################################")
        fake_z2 = self.netG(self.noisev, tch_var_f(self.cam_pos))
        fake_z3 = self.netG(self.noisev, tch_var_f(self.cam_pos))
        # The normal generator is dependent on z
        fake_n2 = self.generate_normals(fake_z2, tch_var_f(self.cam_pos),
                                       self.scene['camera'])
        fake2 = torch.cat([fake_z2, fake_n2], 2)
        fake_recon2, fd_recon, loss = self.render_batch(
            fake2, tch_var_f(self.cam_pos))

        fake_n3 = self.generate_normals(fake_z3, tch_var_f(self.cam_pos),
                                       self.scene['camera'])
        fake3 = torch.cat([fake_z3, fake_n3], 2)
        fake_recon3, fd_recon, loss = self.render_batch(
            fake3, tch_var_f(self.cam_pos))
        # torchvision.utils.save_image(
        #     self.inputv.data, os.path.join(
        #         self.opt.vis_images, 'input_%d.png' % (iteration)),
        #     nrow=2, normalize=True, scale_each=True)
        torchvision.utils.save_image(fake_recon2.data, os.path.join(self.opt.out_dir,  'reconstructed_%d.png' % (iteration)), nrow=2, normalize=True, scale_each=True)
        torchvision.utils.save_image(fake_recon3.data, os.path.join(self.opt.out_dir,  'reconstructed2_%d.png' % (iteration)), nrow=1, normalize=True, scale_each=True)
        noise=(self.noisev[0]).clone().view(1,-1,1,1)
        print(noise[0])
        print("###########################################")
        print(self.cam_pos[0])

        #import ipdb;ipdb.set_trace()
        # noise = Variable(noise)
        fake_z = self.netG(noise,tch_var_f(self.cam_pos[0]).view(1,-1))
        #fake_rendered = self.render_batch(fake,tch_var_f(cam_pos))

        fake_n = self.generate_normals(fake_z, tch_var_f(self.cam_pos[0]).view(1,-1),
                                       self.scene['camera'])
        zz=get_data(noise[0])
        zz2=get_data(self.noisev[0])
        print(zz.shape)
        print(zz2.shape)
        print(np.mean((zz - zz2)**2))
        fake = torch.cat([fake_z, fake_n], 2)
        fake_rendered, fd, loss = self.render_batch(
            fake, tch_var_f(self.cam_pos[0]).view(1,-1))


        torchvision.utils.save_image(fake_rendered.data, os.path.join(self.opt.out_dir,  'test_%d.png' % (iteration)), nrow=1, normalize=True, scale_each=True)

        seq=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        # for i,val in enumerate(seq):
        #     self.iterationa_no = i+2
        #     iteration=i+2
        #     z = val*n1 + (1-val)*self.noisev
        #     fake_z_in = self.netG(z, tch_var_f(self.cam_pos))
        #     # The normal generator is dependent on z
        #     fake_n_in = self.generate_normals(fake_z_in, tch_var_f(self.cam_pos),
        #                                    self.scene['camera'])
        #     fake_in = torch.cat([fake_z_in, fake_n_in], 2)
        #     fake_recon_in, fd_recon_in, loss = self.render_batch(
        #         fake_in, tch_var_f(self.cam_pos))
        #     torchvision.utils.save_image(fake_recon_in.data, os.path.join(self.opt.out_dir,  'in_%d_%d.png' % (i,iteration)), nrow=2, normalize=True, scale_each=True)
        #
        # phi = np.linspace(np.deg2rad(20), np.deg2rad(70), 40)
        # theta = np.ones_like(phi) * np.deg2rad(45)
        # cam_dist_vec = np.ones_like(phi) * self.opt.cam_dist#*0.8
        # cam_pos_1 = np_sph2cart(np.stack((cam_dist_vec, phi, theta), axis=1))
        # theta_2 = np.linspace(np.deg2rad(45), np.deg2rad(75), 30)
        # phi_2 = np.ones_like(theta_2) * np.deg2rad(40)
        # cam_dist_vec_2 = np.ones_like(phi_2) * self.opt.cam_dist#*0.8
        # cam_pos_2 = np_sph2cart(np.stack((cam_dist_vec_2, phi_2, theta_2), axis=1))
        # cam_pos = np.concatenate((cam_pos_1,cam_pos_2))
        #cam_pos = np.split(cam_pos, 100 / 4)
        #for sub_batch in cam_pos:

        #import ipdb;ipdb.set_trace()
        # noise=self.noisev[0].repeat(70,1,1,1)
        #
        # #import ipdb;ipdb.set_trace()
        # # noise = Variable(noise)
        # fake_z = self.netG(noise,tch_var_f(cam_pos))
        # #fake_rendered = self.render_batch(fake,tch_var_f(cam_pos))
        #
        # fake_n = self.generate_normals(fake_z, tch_var_f(cam_pos),
        #                                self.scene['camera'])
        # fake = torch.cat([fake_z, fake_n], 2)
        # fake_rendered, fd, loss = self.render_batch(
        #     fake, tch_var_f(cam_pos))
        #
        #
        # torchvision.utils.save_image(fake_rendered.data, os.path.join(self.opt.out_dir,  'smooth_%d.png' % (iteration)), nrow=10, normalize=True, scale_each=True)
        #




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
    opt = Parameters().parse()

    # Create experiment output folder
    exp_dir = os.path.join(opt.out_dir, opt.name)
    mkdirs(exp_dir)
    sub_dirs=['vis_images','vis_xyz','vis_monitoring']
    for sub_dir in sub_dirs:
        dir_path = os.path.join(exp_dir, sub_dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        setattr(opt, sub_dir, dir_path)


    # Copy scripts to experiment folder
    copy_scripts_to_folder(exp_dir)

    # Save parameters to experiment folder
    file_name = os.path.join(exp_dir, 'opt.txt')
    args = vars(opt)
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

    # Create dataset loader
    dataset_load = Dataset_load(opt)

    # Create GAN
    gan = GAN(opt, dataset_load, exp_dir)

    # Train gan
    gan.train()


if __name__ == '__main__':
    main()
