"""Genrator."""
from __future__ import absolute_import
import os
import sys

from diffrend.generator.generator import D_G_z1

sys.path.append('../../..')
import copy
import numpy as np
from scipy.misc import imsave

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
from parameters import Parameters
from datasets_test import Dataset_load
from networks_test import create_networks

from diffrend.torch.params import SCENE_BASIC
from diffrend.torch.utils import tch_var_f, tch_var_l, where
from diffrend.model import load_model
from diffrend.torch.renderer import render
from diffrend.utils.sample_generator import (uniform_sample_mesh,
                                             uniform_sample_sphere)

from hyperdash import Experiment


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

    def __init__(self, opt, dataset_load=None, experiment=None):
        """Constructor."""
        self.opt = opt
        self.exp = experiment
        self.real_label = 1
        self.fake_label = 0
        self.dataloader = Dataset_load(opt).get_dataloader()

        self.batch_size = opt.batchSize
        # Create dataset loader

        # Create the networks
        self.create_networks()

        # Create create_tensors
        self.create_tensors()

        # Create criterion
        self.create_criterion()

        # Create create optimizers
        self.create_optimizers()

        # Create splats rendering scene

    def create_networks(self, ):
        """Create networks."""
        self.netG, self.netD = create_networks(self.opt, verbose=False)
        if not self.opt.no_cuda:
            self.netD = self.netD.cuda()
            self.netG = self.netG.cuda()

    def create_tensors(self, ):
        """Create the tensors."""
        self.input = torch.FloatTensor(
            self.opt.batchSize, 3, self.opt.imageSize, self.opt.imageSize)
        self.noise = torch.FloatTensor(
            self.opt.batchSize, int(self.opt.nz), 1, 1)
        self.fixed_noise = torch.FloatTensor(
            self.opt.batchSize, int(self.opt.nz)).normal_(0, 1)
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

    def create_criterion(self, ):
        """Create criterion."""
        self.criterion = nn.BCELoss()
        if not self.opt.no_cuda:
            self.criterion = self.criterion.cuda()

    def create_optimizers(self, ):
        """Create optimizers."""
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.opt.lr,
                                     betas=(self.opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.opt.lr,
                                     betas=(self.opt.beta1, 0.999))

    # def get_real_sample(self, i):
    #     """Get a real sample."""
    #     if self.dataset is None:
    #         real_samples = generate_samples(
    #              self.opt.model, self.opt.n_splats, self.opt.splats_radius,
    #              self.opt.width, self.opt.height, self.opt.fovy,
    #              self.opt.focal_length, self.opt.batchSize,
    #              cam_dist=self.cam_dist, cam_pos=self.cam_pos, verbose=False,
    #              obj=None)
    #     else:
    #         if not self.opt.same_view:
    #             self.dataset.set_camera_pos(cam_dist=self.cam_dist,
    #                                         cam_pos=None)
    #         # real_samples = self.data_iter.next()['samples']
    #         real_samples = self.dataset[i]['samples']
    #
    #     self.batch_size = real_samples.size(0)
    #
    #     if not self.opt.no_cuda:
    #         real_samples = real_samples.cuda()
    #
    #     self.input.resize_as_(real_samples.data).copy_(real_samples.data)
    #     self.label.resize_(self.batch_size).fill_(self.real_label)
    #     self.inputv = Variable(self.input)
    #     self.labelv = Variable(self.label)

    def generate_noise_vector(self, ):
        """Generate a noise vector."""
        self.noise.resize_(self.batch_size, int(self.opt.nz)).normal_(0, 1)
        self.noisev = Variable(self.noise)

    def train(self, ):
        """Train networtk."""
        # Start training
        for epoch in range(self.opt.n_iter):
            # self.data_iter = iter(self.dataset_loader)

            for i, data in enumerate(self.dataloader, 0):
                if data[0].size(0) != self.batch_size:
                    continue

                ############################
                # (1) Update D network
                ###########################
                j = 0
                while j < self.opt.critic_iters:
                    j += 1

                    # Train with real
                    self.netD.zero_grad()
                    real_cpu, _ = data
                    batch_size = real_cpu.size(0)
                    if not self.opt.no_cuda:
                        real_cpu = real_cpu.cuda()
                    self.input.resize_as_(real_cpu).copy_(real_cpu)
                    self.label.resize_(self.batch_size).fill_(self.real_label)
                    self.inputv = Variable(self.input)
                    self.labelv = Variable(self.label)

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

                    labelv = Variable(self.label.fill_(self.fake_label))
                    # Do not bp through gen
                    fake_output = self.netD(fake.detach())
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
                            self.netD, self.inputv.data, fake.data,
                            self.opt.gp_lambda)
                        gradient_penalty.backward()
                        errD += gradient_penalty

                    self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                self.generate_noise_vector()
                fake = self.netG(self.noisev)

                # Fake labels are real for generator cost
                labelv = Variable(self.label.fill_(self.real_label))
                fake_output = self.netD(fake)
                if self.opt.criterion == 'WGAN':
                    errG = fake_output.mean()
                    errG.backward(self.mone)
                else:
                    errG = self.criterion(fake_output, labelv)
                    errG.backward()
                D_G_z2 = fake_output.data.mean()
                self.optimizerG.step()

                # Log print
                if i % 50 == 0:
                    print('\n[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f'
                          ' D(G(z)): %.4f / %.4f' % (
                              i, self.opt.n_iter, errD.data[0],
                              errG.data[0], D_x, D_G_z1, D_G_z2))
                    if self.exp is not None:
                        self.exp.metric("epoch", epoch)
                        self.exp.metric("loss D", errD.data[0])
                        self.exp.metric("loss G", errG.data[0])
                        self.exp.metric("D(x)", D_x)
                        self.exp.metric("D(G(z1))", D_G_z1)
                        self.exp.metric("D(G(z2))", D_G_z2)

                # Save images
                if i % 100 == 0:
                    vutils.save_image(real_cpu, '%s/real_samples.png' % self.opt.out_dir,
                                      normalize=True)
                    fake = self.netG(self.fixed_noise)
                    vutils.save_image(
                        fake.data,
                        '%s/fake_samples_epoch_%03d.png' % (self.opt.out_dir, epoch),
                        normalize=True)

                # Do checkpointing
                if i % 500 == 0:
                    self.save_networks(i)

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
    exp = Experiment("diffrend test")

    # Parse args
    opt = Parameters().parse()
    for key, val in opt.__dict__.iter():
        exp.param(key, val)

    # Create dataset loader
    dataset_load = Dataset_load(opt)

    # Create GAN
    gan = GAN(opt, dataset_load, exp)

    # Train gan
    gan.train()


if __name__ == '__main__':
    main()
