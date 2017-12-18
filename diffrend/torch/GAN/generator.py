"""Genrator."""
from __future__ import absolute_import

import copy
import numpy as np
from scipy.misc import imsave

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from parameters import Parameters
# from datasets import Dataset_load
from networks import create_networks

from diffrend.torch.params import SCENE_BASIC
from diffrend.torch.utils import tch_var_f, tch_var_l, where
from diffrend.model import load_model
from diffrend.torch.renderer import render
from diffrend.utils.sample_generator import (uniform_sample_mesh,
                                             uniform_sample_sphere)

# TODO: Parameter
CRITIC_ITERS = 4


def create_scene(width, height, fovy, focal_length, n_samples,
                 radius):
    """Create a semi-empty scene with camera parameters."""
    # Create a splats rendering scene
    scene = copy.deepcopy(SCENE_BASIC)

    # Define the camera parameters
    scene['camera']['viewport'] = [0, 0, width, height]
    scene['camera']['fovy'] = np.deg2rad(fovy)
    scene['camera']['focal_length'] = focal_length
    scene['objects']['disk']['radius'] = tch_var_f(
        np.ones(n_samples) * radius)
    scene['objects']['disk']['material_idx'] = tch_var_l(
        np.zeros(n_samples, dtype=int).tolist())
    scene['materials']['albedo'] = tch_var_f([[0.6, 0.6, 0.6]])
    scene['tonemap']['gamma'] = tch_var_f([1.0])  # Linear output
    return scene


def generate_samples(filename,  n_samples, radius, width, height, fovy,
                     focal_length, batch_size, cam_dist=None, cam_pos=None,
                     verbose=False):
    """Generate random samples of an object from the same camera position.

    Randomly generate N samples on a surface and render them. The samples
    include position and normal, the radius is set to a constant.
    """
    # Check camera Parameters
    if cam_dist is None and cam_pos is None:
        raise ValueError('Use parameter cam_dist or cam_pos')
    elif cam_dist is not None and cam_pos is not None:
        raise ValueError('Use parameter cam_dist or cam_pos. No both.')
    elif cam_dist is not None:
        # generate camera positions on a sphere
        cam_pos = uniform_sample_sphere(radius=cam_dist,
                                        num_samples=batch_size)
        single_view = False
    else:
        single_view = True

    # Load model
    obj = load_model(filename, verbose=verbose)

    # Create a splats rendering scene
    large_scene = create_scene(width, height, fovy, focal_length, n_samples,
                               radius)

    # Generate camera positions on a sphere
    data = []
    for idx in range(batch_size):
        # Sample points from the 3D mesh
        v, vn = uniform_sample_mesh(obj, num_samples=n_samples)

        # Normalize the vertices
        v = (v - np.mean(v, axis=0)) / (v.max() - v.min())

        # Save the splats into the rendering scene
        large_scene['objects']['disk']['pos'] = tch_var_f(v)
        large_scene['objects']['disk']['normal'] = tch_var_f(vn)

        # Set camera position
        if single_view:
            large_scene['camera']['eye'] = tch_var_f(cam_pos)
        else:
            large_scene['camera']['eye'] = tch_var_f(cam_pos[idx])

        # Render scene
        res = render(large_scene)
        depth = res['depth']
        # im = res['image']

        # Normalize depth image
        cond = depth >= large_scene['camera']['far']
        depth = where(cond, torch.min(depth), depth)
        im_depth = ((depth - torch.min(depth)) /
                    (torch.max(depth) - torch.min(depth)))

        # Add depth image to the output structure
        data.append(im_depth.unsqueeze(0))

    return torch.stack(data)


def calc_gradient_penalty(discriminator, real_data, fake_data, gp_lambda):
    """Calculate GP."""
    assert real_data.size(0) == fake_data.size(0)
    alpha = torch.rand(real_data.size(0), 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = Variable(alpha * real_data + ((1 - alpha) * fake_data),
                            requires_grad=True)

    disc_interpolates = discriminator(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda

    return gradient_penalty


class GAN(object):
    """GAN class."""

    def __init__(self, opt):
        """Constructor."""
        self.opt = opt
        self.real_label = 1
        self.fake_label = 0

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

    def create_networks(self,):
        """Create networks."""
        self.netG, self.netD = create_networks(self.opt)
        if not self.opt.no_cuda:
            self.netD = self.netD.cuda()
            self.netG = self.netG.cuda()

    def create_scene(self,):
        """Create a semi-empty scene with camera parameters."""
        self.scene = create_scene(self.opt.width, self.opt.height,
                                  self.opt.fovy, self.opt.f, self.opt.n,
                                  self.opt.r)
        if self.opt.same_view:
            self.cam_pos = uniform_sample_sphere(
                radius=self.opt.cam_dist, num_samples=self.opt.batchSize)

    def create_tensors(self,):
        """Create the tensors."""
        self.input = torch.FloatTensor(
            self.opt.batchSize, 3, self.opt.imageSize, self.opt.imageSize)
        self.noise = torch.FloatTensor(
            self.opt.batchSize, int(self.opt.nz), 1, 1)
        self.fixed_noise = torch.FloatTensor(
            self.opt.batchSize, int(self.opt.nz), 1, 1).normal_(0, 1)
        self.label = torch.FloatTensor(self.opt.batchSize)
        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1

        if not self.opt.no_cuda:
            self.input = self.input.cuda()
            self.label = self.label.cuda()
            self.noise = self.noise.cuda()
            self.fixed_noise = self.fixed_noise.cuda()
            self.one = self.one.cuda()
            self.mone = self.mone.cuda()

        self.fixed_noise = Variable(self.fixed_noise)

    def create_criterion(self,):
        """Create criterion."""
        self.criterion = nn.BCELoss()
        if not self.opt.no_cuda:
            self.criterion = self.criterion.cuda()

    def create_optimizers(self,):
        """Create optimizers."""
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.opt.lr,
                                     betas=(self.opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.opt.lr,
                                     betas=(self.opt.beta1, 0.999))

    def get_real_sample(self,):
        """Get a real sample."""
        if self.opt.same_view:
            cam_dist = None
            cam_pos = self.cam_pos[0]
        else:
            cam_dist = self.opt.cam_dist
            cam_pos = None

        real_sample = generate_samples(
            self.opt.model, self.opt.n, self.opt.r, self.opt.width,
            self.opt.height, self.opt.fovy, self.opt.f, self.opt.batchSize,
            cam_dist=cam_dist, cam_pos=cam_pos, verbose=False)

        self.batch_size = real_sample.size(0)

        if not self.opt.no_cuda:
            real_sample = real_sample.cuda()

        self.input.resize_as_(real_sample.data).copy_(real_sample.data)
        self.label.resize_(self.batch_size).fill_(self.real_label)
        self.inputv = Variable(self.input)
        self.labelv = Variable(self.label)

    def generate_noise_vector(self,):
        """Generate a noise vector."""
        self.noise.resize_(self.batch_size, int(self.opt.nz)).normal_(0, 1)
        self.noisev = Variable(self.noise)

    def render_batch(self, batch, batch_size, scene):
        """Render a batch of splats."""
        # Generate camera positions on a sphere
        if not self.opt.same_view:
            self.cam_pos = uniform_sample_sphere(radius=self.opt.cam_dist,
                                                 num_samples=self.batch_size)

        redered_data = []
        for idx in range(batch_size):
            # Get splats positions and normals
            pos = batch[idx][:, :3]
            pos = (pos - torch.mean(pos, 0)) / (torch.max(pos)-torch.min(pos))
            normals = batch[idx][:, 3:]

            # Set splats into rendering scene
            scene['objects']['disk']['pos'] = pos
            scene['objects']['disk']['normal'] = normals

            # Set camera position
            if not self.opt.same_view:
                scene['camera']['eye'] = tch_var_f(self.cam_pos[idx])
            else:
                scene['camera']['eye'] = tch_var_f(self.cam_pos[0])
            # suffix = '_{}'.format(idx)

            # Render scene
            res = render(scene)

            # Take rendered and depth images
            # im = res['image']
            depth = res['depth']

            # Normalize the depth
            cond = depth >= scene['camera']['far']
            depth = where(cond, torch.min(depth), depth)
            im_depth = ((depth - torch.min(depth)) /
                        (torch.max(depth) - torch.min(depth)))

            # Store normalized depth into the data
            redered_data.append(im_depth.unsqueeze(0))
        redered_data = torch.stack(redered_data)
        return redered_data

    def train(self,):
        """Train networtk."""
        # Start training
        for epoch in range(self.opt.niter):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            for i in range(CRITIC_ITERS):

                # Train with real
                self.netD.zero_grad()
                self.get_real_sample()
                real_output = self.netD(self.inputv)
                if self.opt.criterion == 'GAN':
                    errD_real = self.criterion(real_output, self.labelv)
                    errD_real.backward()
                elif self.opt.criterion == 'WGAN':
                    errD_real = real_output.mean()
                    errD_real.backward(self.mone)
                D_x = real_output.data.mean()

                # Train with fake
                self.generate_noise_vector()
                fake = self.netG(self.noisev)
                fake_rendered = self.render_batch(fake, self.batch_size,
                                                  self.scene)
                labelv = Variable(self.label.fill_(self.fake_label))
                # Do not bp through gen
                fake_output = self.netD(fake_rendered.detach())
                if self.opt.criterion == 'WGAN':
                    errD_fake = fake_output.mean()
                    errD_fake.backward(self.one)
                    errD = errD_fake - errD_real
                else:
                    errD_fake = self.criterion(fake_output, labelv)
                    errD_fake.backward()
                    errD = errD_real + errD_fake
                D_G_z1 = fake_output.data.mean()

                # Compute gradient penalty
                if self.opt.gp != 'None':
                    gradient_penalty = calc_gradient_penalty(
                        self.netD, self.inputv.data, fake_rendered.data,
                        self.opt.gp_lambda)
                    gradient_penalty.backward()
                    errD += gradient_penalty

                self.optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.netG.zero_grad()
            # Fake labels are real for generator cost
            labelv = Variable(self.label.fill_(self.real_label))
            fake_output = self.netD(fake_rendered)
            if self.opt.criterion == 'WGAN':
                errG = fake_output.mean()
                errG.backward(self.mone)
            else:
                errG = self.criterion(fake_output, labelv)
                errG.backward()
            D_G_z2 = fake_output.data.mean()
            self.optimizerG.step()

            # Log print
            if epoch % 10 == 0:
                print('\n[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f'
                      ' D(G(z)): %.4f / %.4f' % (
                          epoch, self.opt.niter, errD.data[0], errG.data[0],
                          D_x, D_G_z1, D_G_z2))

            # Save images
            if epoch % 25 == 0:
                self.save_images(epoch, self.inputv[0], fake_rendered[0])

            # Do checkpointing
            if epoch % 500 == 0:
                self.save_networks(epoch)

            print ("iteration ", epoch, "finished")

    def save_networks(self, epoch):
        """Save networks to hard disk."""
        torch.save(self.netG.state_dict(),
                   '%s/netG_epoch_%d.pth' % (self.opt.out_dir, epoch))
        torch.save(self.netD.state_dict(),
                   '%s/netD_epoch_%d.pth' % (self.opt.out_dir, epoch))

    def save_images(self, epoch, input, output):
        """Save images."""
        imsave(self.opt.out_dir + '/input' + str(epoch) + '.png',
               np.uint8(255. * input.cpu().data.numpy().squeeze()))
        imsave(self.opt.out_dir + '/output' + str(epoch) + '.png',
               np.uint8(255. * output.cpu().data.numpy().squeeze()))


def main():
    """Start training."""
    # Parse args
    opt = Parameters().parse()

    # Load dataset
    # dataloader = Dataset_load(opt).get_dataloader()

    # Create GAN
    gan = GAN(opt)

    # Train gan
    gan.train()


if __name__ == '__main__':
    main()
