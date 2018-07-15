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
            self.opt.axis = None

        # TensorboardX
        self.writer = SummaryWriter(self.opt.vis_monitoring)
        print(self.opt.vis_monitoring)
        print(self.opt.out_dir)
        # Create dataset loader
        self.create_dataset_loader()

        # Create the networks


        # Create create_tensors
        self.create_tensors()

        # Create criterion


        # Create create optimizers


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
        if not self.opt.same_view:
            if self.opt.full_sphere_sampling:
                self.cam_pos = uniform_sample_sphere(
                    radius=self.opt.cam_dist, num_samples=self.opt.batchSize,
                    axis=self.opt.axis, angle=np.deg2rad(self.opt.angle),
                    theta_range=self.opt.theta, phi_range=self.opt.phi)
            else:
                self.cam_pos = uniform_sample_sphere(
                    radius=self.opt.cam_dist, num_samples=self.opt.batchSize,
                    axis=self.opt.axis, angle=self.opt.angle,
                    theta_range=np.deg2rad(self.opt.theta),
                    phi_range=np.deg2rad(self.opt.phi))
        if  self.opt.full_sphere_sampling_light:
            self.light_pos1 = uniform_sample_sphere(radius=self.opt.cam_dist, num_samples=self.opt.batchSize,
                                                 axis=self.opt.axis, angle=np.deg2rad(44),
                                                 theta_range=self.opt.theta, phi_range=self.opt.phi)
            # self.light_pos2 = uniform_sample_sphere(radius=self.opt.cam_dist, num_samples=self.opt.batchSize,
            #                                      axis=self.opt.axis, angle=np.deg2rad(40),
            #                                      theta_range=self.opt.theta, phi_range=self.opt.phi)
        else:
            print("inbox")
            light_eps = 0.15
            self.light_pos1 = np.random.rand(self.opt.batchSize,3)*self.opt.cam_dist + light_eps
            self.light_pos2 = np.random.rand(self.opt.batchSize,3)*self.opt.cam_dist + light_eps

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
        inpath2 = self.opt.vis_input + '/'
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
            #large_scene['lights']['pos'][1,:3]=tch_var_f(self.light_pos2[idx])

            # Render scene
            res = render(large_scene,
                         norm_depth_image_only=self.opt.norm_depth_image_only,
                         double_sided=True, use_quartic=self.opt.use_quartic, tile_size=1024)

            # Get rendered output
            if self.opt.render_img_nc == 1:
                depth = res['depth']
                im_d = depth.unsqueeze(0)
            else:
                depth = res['depth']
                im_d = depth.unsqueeze(0)
                im = res['image'].permute(2, 0, 1)
                im_ = get_data(res['image'])
                #im_img_ = get_normalmap_image(im_)
                target_normal_ = get_data(res['normal'])
                target_normalmap_img_ = get_normalmap_image(target_normal_)
                im_n = tch_var_f(
                    target_normalmap_img_).view(im.shape[1], im.shape[2],
                                                3).permute(2, 0, 1)

            # Add depth image to the output structure
            file_name = inpath2 + str(self.iterationa_no) +"_"+str(self.critic_iter)+'input_{:05d}.txt'.format(idx)
            text_file = open(file_name, "w")
            text_file.write('%s\n' % (str(large_scene['camera']['eye'].data)))
            text_file.close()
            out_file_name = inpath2 + str(self.iterationa_no) +"_"+str(self.critic_iter)+'input_{:05d}.npy'.format(idx)
            np.save(out_file_name, self.cam_pos[idx])
            out_file_name2 = inpath2 + str(self.iterationa_no) +"_"+str(self.critic_iter)+'input_light{:05d}.npy'.format(idx)
            np.save(out_file_name2, self.light_pos1[idx])
            imsave((inpath2 + str(self.iterationa_no) +"_"+str(self.critic_iter)+
                    'input_{:05d}.png'.format(idx)),
                   im_)
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
            self.batch_size, int(self.opt.nz), 1, 1).normal_(0, 1)
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
                               get_data(torch.mean(torch.abs(grad[:,:,0]))),
                               self.iterationa_no)
        self.writer.add_scalar("position_gradient_mean_channel2",
                               get_data(torch.mean(torch.abs(grad[:,:,1]))),
                               self.iterationa_no)
        self.writer.add_scalar("position_gradient_mean_channel3",
                               get_data(torch.mean(torch.abs(grad[:,:,2]))),
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
                               get_data(torch.mean(torch.abs(grad[:,:,0]))),
                               self.iterationa_no)
        self.writer.add_scalar("normal_gradient_mean_channel2",
                               get_data(torch.mean(torch.abs(grad[:,:,1]))),
                               self.iterationa_no)
        self.writer.add_scalar("normal_gradient_mean_channel3",
                               get_data(torch.mean(torch.abs(grad[:,:,2]))),
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
                               get_data(torch.mean(torch.abs(grad))),
                               self.iterationa_no)
        self.writer.add_histogram("z_gradient_hist_channel", grad.clone().cpu().data.numpy(),self.iterationa_no)

        self.writer.add_image("z_gradient_im",
                               grad,
                               self.iterationa_no)


    
    def tensorboard_hook(self, grad):
        self.writer.add_scalar("z_gradient_mean",
                               get_data(torch.mean(grad[0])),
                               self.iterationa_no)
        self.writer.add_histogram("z_gradient_hist_channel", grad[0].clone().cpu().data.numpy(),self.iterationa_no)

        self.writer.add_image("z_gradient_im",
                               grad[0].view(self.opt.splats_img_size,self.opt.splats_img_size),
                               self.iterationa_no)

    def train(self, ):
        """Train network."""
        # Load pretrained model if required
        if self.opt.gen_model_path is not None:
            print("Reloading networks from")
            print(' > Generator', self.opt.gen_model_path)
            self.netG.load_state_dict(
                torch.load(open(self.opt.gen_model_path, 'rb')))
            print(' > Generator2', self.opt.gen_model_path2)
            self.netG2.load_state_dict(
                torch.load(open(self.opt.gen_model_path2, 'rb')))
            print(' > Discriminator', self.opt.dis_model_path)
            self.netD.load_state_dict(
                torch.load(open(self.opt.dis_model_path, 'rb')))
            print(' > Discriminator2', self.opt.dis_model_path2)
            self.netD2.load_state_dict(
                torch.load(open(self.opt.dis_model_path2, 'rb')))

        # Start training
        file_name = os.path.join(self.opt.out_dir, 'L2.txt')
        with open(file_name, 'wt') as l2_file:
            curr_generator_idx = 0
            for iteration in range(self.opt.n_iter):
                self.iterationa_no = iteration
                self.critic_iter=0
                # Train Discriminator critic_iters times
                for j in range(self.opt.critic_iters):
                    # Train with real
                    #################
                    self.in_critic=1

                    self.get_real_samples()
                    self.critic_iter+=1


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
    sub_dirs=['vis_images','vis_xyz','vis_monitoring','vis_input']
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
