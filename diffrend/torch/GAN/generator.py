from __future__ import absolute_import

import os, sys
sys.path.append('../..')
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from parameters import Parameters
from datasets import Dataset_load
from networks import create_networks
from utils import where
from diffrend.torch.params import SCENE_BASIC
from diffrend.torch.utils import tch_var_f, tch_var_l, CUDA
from diffrend.torch.renderer import render
from diffrend.utils.sample_generator import uniform_sample_mesh, uniform_sample_sphere
from diffrend.model import load_model
from data import DIR_DATA
import argparse
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave

# Get parameters
parser = argparse.ArgumentParser(usage="splat_gen_render_demo.py --model filename --out_dir output_dir "
                                       "--n 5000 --width 128 --height 128 --r 0.025 --cam_dist 5 --nv 10")
parser.add_argument('--model', type=str, default=DIR_DATA + '/chair_0001.off')
parser.add_argument('--out_dir', type=str, default='./render_samples/')
parser.add_argument('--width', type=int, default=128)
parser.add_argument('--height', type=int, default=128)
parser.add_argument('--n', type=int, default=5000)
parser.add_argument('--r', type=float, default=0.025)
parser.add_argument('--cam_dist', type=float, default=5.0, help='Camera distance from the center of the object')
parser.add_argument('--nv', type=int, default=10, help='Number of views to generate')
parser.add_argument('--fovy', type=float, default=15.0, help='Field of view in the vertical direction')
parser.add_argument('--f', type=float, default=0.1, help='focal length')

args = parser.parse_args()

def same(filename,  num_samples, radius, width, height,fovy, focal_length,cam_pos,batch_size):
    """
    Randomly generate N samples on a surface and render them. The samples include position and normal, the radius is set
    to a constant.
    """
    obj = load_model(filename)


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



    data=[]
    for idx in range(batch_size):
        v, vn = uniform_sample_mesh(obj, num_samples=num_samples)

        # normalize the vertices
        v = (v - np.mean(v, axis=0)) / (v.max() - v.min())

        large_scene['objects']['disk']['pos'] = tch_var_f(v)
        large_scene['objects']['disk']['normal'] = tch_var_f(vn)

        large_scene['camera']['eye'] = tch_var_f(cam_pos)
        suffix = '_{}'.format(idx)

        # main render run
        res = render(large_scene)
        if CUDA:
            im = res['image']
        else:
            im = res['image']

        if CUDA:
            depth = res['depth']
        else:
            depth = res['depth']
        #import ipdb; ipdb.set_trace()
        cond = depth >= large_scene['camera']['far']
        depth = where(cond, torch.min(depth), depth)
        #depth[depth >= large_scene['camera']['far']] = torch.min(depth)
        im_depth =(depth - torch.min(depth)) / (torch.max(depth) - torch.min(depth))

        data.append(im_depth.unsqueeze(0))
    return torch.stack(data)




print(args)
if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

opt = Parameters().parse()

# Load dataset
#dataloader = Dataset_load(opt).get_dataloader()

# Create the networks
netG, netD = create_networks(opt,args)

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

# Setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
cam_pos = uniform_sample_sphere(radius=args.cam_dist, num_samples=2)
for epoch in range(opt.niter):

    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # train with real
    netD.zero_grad()
    real_cpu = same(args.model, args.n, args.r,  args.width, args.height, args.fovy, args.f,cam_pos[0],opt.batchSize)


    batch_size = real_cpu.size(0)
    if not opt.no_cuda:
        real_cpu = real_cpu.cuda()
    input.resize_as_(real_cpu.data).copy_(real_cpu.data)
    label.resize_(batch_size).fill_(real_label)
    inputv = Variable(input)
    labelv = Variable(label)

    output = netD(inputv)
    errD_real = criterion(output, labelv)
    errD_real.backward()
    D_x = output.data.mean()

    # train with fake
    noise.resize_(batch_size, int(opt.nz), 1, 1).normal_(0, 1)
    noisev = Variable(noise)
    fake = netG(noisev)
    #######################
    #processig generator output to get image
    ########################
    r = np.ones(args.n) * args.r
    large_scene = copy.deepcopy(SCENE_BASIC)
    large_scene['camera']['viewport'] = [0, 0, args.width,args.height]
    large_scene['camera']['fovy'] = np.deg2rad(args.fovy)
    large_scene['camera']['focal_length'] =args.f
    large_scene['objects']['disk']['radius'] = tch_var_f(r)
    large_scene['objects']['disk']['material_idx'] = tch_var_l(np.zeros(args.n, dtype=int).tolist())
    large_scene['materials']['albedo'] = tch_var_f([[0.6, 0.6, 0.6]])
    large_scene['tonemap']['gamma'] = tch_var_f([1.0])  # Linear output

    # generate camera positions on a sphere

    data=[]
    for idx in range(batch_size):

        #import ipdb; ipdb.set_trace()
        # normalize the vertices
        temp = (fake[idx][:, :3] - torch.mean(fake[idx][:, :3], 0))/(torch.max(fake[idx][:, :3]) - torch.min(fake[idx][:, :3]))

        large_scene['objects']['disk']['pos'] = temp
        large_scene['objects']['disk']['normal'] = fake[idx][:, 3:]

        large_scene['camera']['eye'] = tch_var_f(cam_pos[0])
        suffix = '_{}'.format(idx)

        # main render run
        res = render(large_scene)
        if CUDA:
            im = res['image']
        else:
            im = res['image']

        if CUDA:
            depth = res['depth']
        else:
            depth = res['depth']

        cond = depth >= large_scene['camera']['far']
        depth = where(cond, torch.min(depth), depth)
        im_depth =(depth - torch.min(depth)) / (torch.max(depth) - torch.min(depth))
        data.append(im_depth.unsqueeze(0))


    data=torch.stack(data)
    labelv = Variable(label.fill_(fake_label))
    output = netD(data.detach())  # Do not backpropagate through generator
    errD_fake = criterion(output, labelv)
    errD_fake.backward()
    D_G_z1 = output.data.mean()
    errD = errD_real + errD_fake
    optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    # Fake labels are real for generator cost
    labelv = Variable(label.fill_(real_label))
    output = netD(data)
    errG = criterion(output, labelv)
    errG.backward()
    D_G_z2 = output.data.mean()
    optimizerG.step()
    suffix = '_{}'.format(epoch)
    if epoch % 10 == 0:
        print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f'
              ' D(G(z)): %.4f / %.4f' % (epoch, opt.niter,

                                         errD.data[0], errG.data[0], D_x,
                                         D_G_z1, D_G_z2), end="\r")
    if epoch % 100 == 0:
        imsave(args.out_dir + '/img' + suffix + '.png', np.uint8(255. *real_cpu[0].cpu().data.numpy().squeeze()))
        imsave(args.out_dir + '/img_depth' + suffix + '.png', np.uint8(255. * data[0].cpu().data.numpy().squeeze()))
        imsave(args.out_dir + '/img1' + suffix + '.png', np.uint8(255. *real_cpu[1].cpu().data.numpy().squeeze()))
        imsave(args.out_dir + '/img_depth1' + suffix + '.png', np.uint8(255. * data[1].cpu().data.numpy().squeeze()))


    # Do checkpointing
    if epoch % 100 == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
        print ("iteration ", epoch, "finished")
