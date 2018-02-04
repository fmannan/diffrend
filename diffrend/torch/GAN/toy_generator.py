"""Genrator."""
from __future__ import absolute_import

import os
import sys
sys.path.append('../../..')
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from parameters import Parameters
from datasets import Dataset_load
from toy_networks import create_networks
from utils import where
from diffrend.torch.params import SCENE_BASIC
from diffrend.torch.utils import tch_var_f, tch_var_l, CUDA, get_data, normalize
from diffrend.torch.renderer import render
from diffrend.utils.sample_generator import uniform_sample_mesh, uniform_sample_sphere
from diffrend.model import load_model
# from data import DIR_DATA
import argparse
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
import torch.nn.functional as F

from data import DIR_DATA
CRITIC_ITERS=4





# TODO: This function is the same as the previous one except for one line. Can
# we add a parameter and combine both?
def different_views(filename, num_samples, radius, cam_dist,  width, height,
                    fovy, focal_length, batch_size, verbose=False):
    """Generate rendom samples of an object from different camera positions.

    Randomly generate N samples on a surface and render them. The samples
    include position and normal, the radius is set to a constant.
    """



    fovy=11.5
    num_samples = width * height
    r = np.ones(num_samples) * radius

    large_scene = copy.deepcopy(SCENE_BASIC)

    large_scene['camera']['viewport'] = [0, 0, width, height]
    large_scene['camera']['fovy'] = np.deg2rad(fovy)
    large_scene['camera']['focal_length'] = focal_length
    large_scene['objects']['disk']['radius'] = tch_var_f(r)
    large_scene['objects']['disk']['material_idx'] = tch_var_l(np.zeros(num_samples, dtype=int).tolist())
    large_scene['materials']['albedo'] = tch_var_f([[0.6, 0.6, 0.6]])
    large_scene['tonemap']['gamma'] = tch_var_f([1.0])  # Linear output

    # generate camera positions on a sphere
    cam_pos =   [0, 0, 10]

    data = []
    for idx in range(batch_size):
        x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
        #z = np.sqrt(1 - np.min(np.stack((x ** 2 + y ** 2, np.ones_like(x)), axis=-1), axis=-1))
        unit_disk_mask = (x ** 2 + y ** 2) <= 1
        z = np.sqrt(1 - unit_disk_mask * (x ** 2 + y ** 2))

        # Make a hemi-sphere bulging out of the xy-plane scene
        z[~unit_disk_mask] = 0
        z_old=z
        z=z+np.random.normal(0,0.01,z.shape)
        pos = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)

        # Normals outside the sphere should be [0, 0, 1]
        x[~unit_disk_mask] = 0
        y[~unit_disk_mask] = 0
        z_old[~unit_disk_mask] = 1

        normals = np.stack((x.ravel(), y.ravel(), z_old.ravel()), axis=1)
        norm = np.sqrt(np.sum(normals ** 2, axis=1))
        normals = normals / norm[..., np.newaxis]
        normals=normals+np.random.normal(0,0.01,normals.shape)


        large_scene['objects']['disk']['pos'] = tch_var_f(pos)
        large_scene['objects']['disk']['normal'] = tch_var_f(normals)

        large_scene['camera']['eye'] = tch_var_f(cam_pos)

        res = render(large_scene)

        im =res['image']
        depth = res['depth']

        # import ipdb; ipdb.set_trace()

        # Normalize depth image


        # Add depth image to the output structure
        data.append(im.unsqueeze(0))

        #import ipdb; ipdb.set_trace()
    return torch.stack(data)




############################
# MAIN
###########################
# TODO: Better move to a train function and create an entry point

# Parse args
opt = Parameters(DIR_DATA).parse()
def calc_gradient_penalty(discriminator, real_data, fake_data):
    """Calculate GP."""
    assert real_data.size(0) == fake_data.size(0)
    alpha = torch.rand(real_data.size(0), 1, 1,1)
    # alpha = torch.rand(real_data.size(0), 1)
    # alpha = alpha.expand(real_data.size(0), real_data.nelement()/real_data.size(0)).contiguous().view(real_data.size(0),real_data.size(1), real_data.size(2), real_data.size(3))
    #
    alpha = alpha.expand(real_data.size())
    #  import ipdb; ipdb.set_trace()
    alpha = alpha.cuda()

    interpolates = Variable(
        alpha * real_data + ((1 - alpha) * fake_data),
        requires_grad=True
    )

    disc_interpolates = discriminator(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * \
        opt.gp_lambda

    return gradient_penalty
# Load dataset
# dataloader = Dataset_load(opt).get_dataloader()

# Create the networks
netG, netD = create_networks(opt)

# Create the criterion
criterion = nn.BCELoss()

# Create the tensors
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, int(opt.nz), 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, int(opt.nz), 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

# Move everything to the GPU
if not opt.no_cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)
one = torch.FloatTensor([1])
mone = one * -1
if not opt.no_cuda:
    one = one.cuda()
    mone = mone.cuda()
# Setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# Create splats rendering scene
fovy=11.5
num_samples = opt.width * opt.height
r = np.ones(num_samples) * opt.r
large_scene = copy.deepcopy(SCENE_BASIC)

large_scene['camera']['viewport'] = [0, 0, opt.width, opt.height]
large_scene['camera']['fovy'] = np.deg2rad(fovy)
large_scene['camera']['focal_length'] = opt.f
large_scene['objects']['disk']['radius'] = tch_var_f(r)
large_scene['objects']['disk']['material_idx'] = tch_var_l(np.zeros(num_samples, dtype=int).tolist())
large_scene['materials']['albedo'] = tch_var_f([[0.6, 0.6, 0.6]])
large_scene['tonemap']['gamma'] = tch_var_f([1.0])  # Linear output
# large_scene['camera']['eye'] = tch_var_f(cam_pos[0])

# Start training
for epoch in range(opt.niter):

    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # train with real

    for i in range(CRITIC_ITERS):
        netD.zero_grad()
        if opt.same_view:
            real_cpu = same_view(opt.model, opt.n, opt.r,  opt.width,
                                 opt.height, opt.fovy, opt.f, np.copy(cam_pos),
                                 opt.batchSize)
        else:
            real_cpu = different_views(opt.model, opt.n, opt.r, opt.cam_dist,
                                       opt.width, opt.height, opt.fovy, opt.f,
                                       opt.batchSize)
        batch_size = real_cpu.size(0)
        real_cpu=real_cpu.squeeze()

        if not opt.no_cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu.data).copy_(real_cpu.data)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        real_output = netD(inputv.permute(0, 3, 1, 2))
        if opt.criterion == 'GAN':
            #print('BCE targets: %.4f Loss_G: %.4f' % (real_output.data[0], labelv.data[0]))
            errD_real = criterion(real_output, labelv)
            errD_real.backward()
        if opt.criterion == 'WGAN':
            errD_real = real_output.mean()
            errD_real.backward(mone)

        D_x = real_output.data.mean()

        # train with fake
        noise.resize_(batch_size, int(opt.nz)).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        fake_normals_norm = torch.sqrt(torch.sum(fake[:, :, 1:] *fake[:, :, 1:] , dim=-1))
        # #print(fake_normals_norm.size(), fake_normals.size())
        fake_normals = fake[:, :, 1:] / fake_normals_norm[:, :,  np.newaxis]
        #######################
        #processig generator output to get image
        ########################


        # generate camera positions on a sphere

        data=[]
        pos_array=[]
        normal_array=[]
        #cam_pos = uniform_sample_sphere(radius=args.cam_dist, num_samples=batch_size)
        if not opt.same_view:
            cam_pos = [0, 0, 10]
        #import ipdb; ipdb.set_trace()
        for idx in range(batch_size):

            #import ipdb; ipdb.set_trace()
            # normalize the vertices
            x, y = np.meshgrid(np.linspace(-1, 1,  opt.width), np.linspace(-1, 1, opt.height))
            pos = np.stack((x.ravel(), y.ravel()), axis=1)
            pos = tch_var_f(pos)
            temp =torch.cat([pos, fake[idx][:, :1]], 1)
            #temp = (fake[idx][:, :3] - torch.mean(fake[idx][:, :3], 0))/(torch.max(fake[idx][:, :3]) - torch.min(fake[idx][:, :3]))

            large_scene['objects']['disk']['pos'] = temp
            large_scene['objects']['disk']['normal'] = fake[idx][:, 1:]
            #large_scene['camera']['eye'] = tch_var_f(cam_pos[idx])

            large_scene['camera']['eye'] = tch_var_f(cam_pos)


            suffix = '_{}'.format(idx)

            # main render run
            res = render(large_scene)
            im = res['image']
            depth = res['depth']


            data.append(im)
            pos_array.append(temp)
            normal_array.append(fake[idx][:, 1:])

        eye=tch_var_f(cam_pos)
        data=torch.stack(data)
        normal_array=torch.stack(normal_array)
        pos_array=torch.stack(pos_array)
        #import ipdb; ipdb.set_trace()
        cam_dir=normalize(eye[np.newaxis, np.newaxis, :] - pos_array)

        towards_cam_loss=torch.mean(torch.sum( F.relu(-torch.sum(normal_array * cam_dir, dim=-1)), dim=-1))

        gray_img = torch.mean(data, dim=-1)

        dx = gray_img[:, :, 1:] - gray_img[:, :, :-1]
        dy = gray_img[:, 1:, :] - gray_img[:, :-1, :]
        grad_img = torch.abs(dx) + torch.abs(dy)
        image_gradient_loss = torch.mean(grad_img)
        #data=data.squeeze()

        labelv = Variable(label.fill_(fake_label))
        fake_output = netD(data.permute(0, 3, 1, 2).detach())  # Do not backpropagate through generator
        if opt.criterion == 'WGAN':
            errD_fake = fake_output.mean()

            errD_fake.backward(one)
            errD=errD_fake-errD_real
        else:
            errD_fake = criterion(fake_output, labelv)
            errD_fake.backward()
            errD = errD_real + errD_fake

        D_G_z1 =fake_output.data.mean()

        if opt.gp != 'None':

            gradient_penalty = calc_gradient_penalty(
            netD, inputv.permute(0, 3, 1, 2).data, data.permute(0, 3, 1, 2).data
            )
            gradient_penalty.backward()
            errD += gradient_penalty


        optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################

    netG.zero_grad()

    #Fake labels are real for generator cost
    labelv = Variable(label.fill_(real_label))
    fake_output = netD(data.permute(0, 3, 1, 2))
    if opt.criterion == 'WGAN':
        errG = fake_output.mean()
        errG+= opt.toward_cam * towards_cam_loss + opt.im_grad * image_gradient_loss
        errG.backward(mone)
    else:
        errG = criterion(fake_output, labelv)
        errG.backward()
    D_G_z2 = fake_output.data.mean()
    optimizerG.step()
    suffix = '_{}'.format(epoch)
    if epoch % 10 == 0:
        print('\n[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f'
              ' D(G(z)): %.4f / %.4f' % (epoch, opt.niter,

                                         errD.data[0], errG.data[0], D_x,
                                         D_G_z1, D_G_z2))

    if epoch % 25 == 0:
        imsave(opt.out_dir + '/img' + suffix + '.png',
               np.uint8(255. * real_cpu[0].cpu().data.numpy().squeeze()))
        imsave(opt.out_dir + '/img_depth' + suffix + '.png',
               np.uint8(255. * data[0].cpu().data.numpy().squeeze()))
        imsave(opt.out_dir + '/img1' + suffix + '.png',
               np.uint8(255. * real_cpu[1].cpu().data.numpy().squeeze()))
        imsave(opt.out_dir + '/img_depth1' + suffix + '.png',
               np.uint8(255. * data[1].cpu().data.numpy().squeeze()))

    # Do checkpointing
    if epoch % 100 == 0:
        torch.save(netG.state_dict(),
                   '%s/netG_epoch_%d.pth' % (opt.out_dir, epoch))
        torch.save(netD.state_dict(),
                   '%s/netD_epoch_%d.pth' % (opt.out_dir, epoch))
        print ("iteration ", epoch, "finished")
