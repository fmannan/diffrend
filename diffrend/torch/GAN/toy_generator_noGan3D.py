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
from diffrend.torch.utils import tch_var_f, tch_var_l, CUDA, get_data
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
    data=[]
    pos_array=[]
    normals_array=[]
    for idx in range(batch_size):
        x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
        #z = np.sqrt(1 - np.min(np.stack((x ** 2 + y ** 2, np.ones_like(x)), axis=-1), axis=-1))
        unit_disk_mask = (x ** 2 + y ** 2) <= 1
        z = np.sqrt(1 - unit_disk_mask * (x ** 2 + y ** 2))

        # Make a hemi-sphere bulging out of the xy-plane scene
        z[~unit_disk_mask] = 0
        z_old=z
        pos = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
        # Normals outside the sphere should be [0, 0, 1]
        x[~unit_disk_mask] = 0
        y[~unit_disk_mask] = 0
        z_old[~unit_disk_mask] = 1

        normals = np.stack((x.ravel(), y.ravel(), z_old.ravel()), axis=1)
        norm = np.sqrt(np.sum(normals ** 2, axis=1))
        normals = normals / norm[..., np.newaxis]
        large_scene['objects']['disk']['pos'] = tch_var_f(pos)
        large_scene['objects']['disk']['normal'] = tch_var_f(normals)
        large_scene['camera']['eye'] = tch_var_f(cam_pos)
        res = render(large_scene)
        im =res['image']
        depth = res['depth']
        # Normalize depth image
        cond = depth >= large_scene['camera']['far']
        depth = where(cond, torch.min(depth), depth)
        # depth[depth >= large_scene['camera']['far']] = torch.min(depth)
        im_depth = ((depth - torch.min(depth)) /
                    (torch.max(depth) - torch.min(depth)))

        # Add depth image to the output structure
        data.append(im_depth.unsqueeze(0))
        pos_array.append(tch_var_f(pos).unsqueeze(0))
        normals_array.append(tch_var_f(normals).unsqueeze(0))


    return torch.stack(data),torch.stack(pos_array),torch.stack(normals_array)




############################
# MAIN
###########################
# TODO: Better move to a train function and create an entry point

# Parse args
opt = Parameters(DIR_DATA).parse()

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
#optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
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
    real_cpu, real_pos, real_normals = different_views(opt.model, opt.n, opt.r, opt.cam_dist,
                                   opt.width, opt.height, opt.fovy, opt.f,
                                   opt.batchSize)

    batch_size = real_cpu.size(0)
    if not opt.no_cuda:
        real_cpu = real_cpu.cuda()
    input.resize_as_(real_cpu.data).copy_(real_cpu.data)
    # label.resize_(batch_size).fill_(real_label)
    inputv = Variable(input)
    noise.resize_(batch_size, int(opt.nz)).normal_(0, 1)
    noisev = Variable(noise)
    fake = netG(noisev)
    fake_normals_norm = torch.sqrt(torch.sum(fake[:, :,1:] *fake[:, :,1:] , dim=-1))
    #print(fake_normals_norm.size(), fake_normals.size())
    fake_normals = fake[:, :,1:] / fake_normals_norm[:, :, :, np.newaxis]
    #######################
    #processig generator output to get image
    ########################

    data=[]
    cam_pos = [0, 0, 10]

    for idx in range(batch_size):
        # normalize the vertices
        x, y = np.meshgrid(np.linspace(-1, 1,  opt.width), np.linspace(-1, 1, opt.height))
        pos = np.stack((x.ravel(), y.ravel()), axis=1)
        pos = tch_var_f(pos)
        temp =torch.cat([pos, fake[idx][:, :1]], 1)
        #temp = (fake[idx][:, :3] - torch.mean(fake[idx][:, :3], 0))/(torch.max(fake[idx][:, :3]) - torch.min(fake[idx][:, :3]))

        large_scene['objects']['disk']['pos'] = temp
        # fake_normals_norm = torch.sqrt(torch.sum(fake[idx][:, 1:] * fake[idx][:, 1:], dim=-1))
        # #print(fake_normals_norm.size(), fake_normals.size())
        # fake[idx][:, 1:] = fake[idx][:, 1:] / fake_normals_norm[ :, :, np.newaxis]
        large_scene['objects']['disk']['normal'] = fake[idx][:, 1:]
        temp_normal=fake[idx][:, 1:]
        #large_scene['camera']['eye'] = tch_var_f(cam_pos[idx])

        large_scene['camera']['eye'] = tch_var_f(cam_pos)
        suffix = '_{}'.format(idx)
        # main render run
        res = render(large_scene)
        im = res['image']
        depth = res['depth']

        cond = depth >= large_scene['camera']['far']
        depth = where(cond, torch.min(depth), depth)
        im_depth =(depth - torch.min(depth)) / (torch.max(depth) - torch.min(depth))
        data.append(im_depth.unsqueeze(0))




    data=torch.stack(data)
    #import ipdb; ipdb.set_trace()



    mse_criterion = nn.MSELoss().cuda()
    gen_loss=mse_criterion(data, real_cpu)

    netG.zero_grad()


    gen_loss.backward()

    optimizerG.step()
    suffix = '_{}'.format(epoch)
    if epoch % 10 == 0:
        print('\n[%d/%d]Loss_G: %.4f' % (epoch, opt.niter,gen_loss.data[0]))

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
