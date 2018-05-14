from diffrend.torch.params import SCENE_BASIC, SCENE_1, SCENE_2
from diffrend.torch.renderer import render, render_splats_NDC, render_splats_along_ray, z_to_pcl_CC
from diffrend.torch.utils import (tch_var_f, tch_var_l, CUDA, get_data, get_normalmap_image,
                                  world_to_cam, cam_to_world, normalize, unit_norm2_L2loss,
                                  normalize_maxmin, normal_consistency_cost, away_from_camera_penalty,
                                  spatial_3x3)
from diffrend.torch.ops import sph2cart_unit
from diffrend.utils.utils import save_xyz
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
from diffrend.torch.NEstNet import NEstNet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import argparse


def render_scene(scene, output_folder, norm_depth_image_only=False, backface_culling=False, plot_res=True):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # main render run
    res = render(scene, norm_depth_image_only=norm_depth_image_only, backface_culling=backface_culling)
    im = get_data(res['image'])
    im_nearest = get_data(res['nearest'])
    obj_pixel_count = get_data(res['obj_pixel_count']) if 'obj_pixel_count' in res else None

    if plot_res:
        plt.ion()
        plt.figure()
        plt.imshow(im)
        plt.title('Final Rendered Image')
        plt.savefig(output_folder + '/img_torch.png')

        plt.figure()
        plt.imshow(im_nearest)
        plt.title('Nearest Object Index')
        plt.colorbar()
        plt.savefig(output_folder + '/img_nearest.png')

        plt.figure()
        plt.plot(obj_pixel_count, 'r-+')
        plt.xlabel('Object Index')
        plt.ylabel('Number of Pixels')

    depth = get_data(res['depth'])
    depth[depth >= scene['camera']['far']] = np.inf
    print(depth.min(), depth.max())
    if plot_res and depth.min() != np.inf:
        plt.figure()
        plt.imshow(depth)
        plt.title('Depth Image')
        plt.savefig(output_folder + '/img_depth_torch.png')

    if plot_res:
        plt.ioff()
        plt.show()

    return res


def optimize_scene(input_scene, target_scene, out_dir, max_iter=100, lr=1e-3, print_interval=10,
                   imsave_interval=10):
    """A demo function to check if the differentiable renderer can optimize.
    :param scene:
    :param out_dir:
    :return:
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    target_res = render(target_scene)
    target_im = target_res['image']
    target_im.require_grad = False
    criterion = nn.MSELoss()
    if CUDA:
        target_im_ = target_res['image'].cpu()
        criterion = criterion.cuda()

    plt.ion()
    plt.figure()
    plt.imshow(target_im_.data.numpy())
    plt.title('Target Image')
    plt.savefig(out_dir + 'target.png')

    input_scene['materials']['albedo'].requires_grad = True
    optimizer = optim.Adam(input_scene['materials'].values(), lr=lr)

    h0 = plt.figure()
    h1 = plt.figure()
    loss_per_iter = []
    for iter in range(max_iter):
        res = render(input_scene)
        im_out = res['image']

        optimizer.zero_grad()
        loss = criterion(im_out, target_im)

        im_out_ = get_data(im_out)
        loss_ = get_data(loss)
        loss_per_iter.append(loss_)

        if iter == 0:
            plt.figure(h0.number)
            plt.imshow(im_out_)
            plt.title('Initial')

        if iter % print_interval == 0:
            print('%d. loss= %f' % (iter, loss_))
            print(input_scene['materials'])

            plt.figure(h1.number)
            plt.imshow(im_out_)
            plt.title('%d. loss= %f' % (iter, loss_))
            plt.savefig(out_dir + '/fig_%05d.png' % iter)

        loss.backward()
        optimizer.step()

    plt.figure()
    plt.plot(loss_per_iter, linewidth=2)
    plt.xlabel('Iteration', fontsize=14)
    plt.title('MSE Loss', fontsize=12)
    plt.grid(True)
    plt.savefig(out_dir + '/loss.png')

    plt.ioff()
    plt.show()


def optimize_NDC_test(out_dir, width=32, height=32, max_iter=100, lr=1e-3, scale=10, print_interval=10,
                      imsave_interval=10):
    """A demo function to check if the differentiable renderer can optimize splats in NDC.
    :param scene:
    :param out_dir:
    :return:
    """
    import torch
    import copy
    from diffrend.torch.params import SCENE_SPHERE_HALFBOX

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    scene = SCENE_SPHERE_HALFBOX
    scene['camera']['viewport'] = [0, 0, width, height]
    scene['camera']['fovy'] = np.deg2rad(45)
    scene['camera']['focal_length'] = 1
    scene['camera']['eye'] = tch_var_f([2, 1, 2, 1])
    scene['camera']['at'] = tch_var_f([0, 0.8, 0, 1])

    target_res = render(SCENE_SPHERE_HALFBOX)
    target_im = target_res['image']
    target_im.require_grad = False
    target_im_ = get_data(target_res['image'])

    criterion = nn.L1Loss() #nn.MSELoss()
    criterion = criterion.cuda()

    plt.ion()
    plt.figure()
    plt.imshow(target_im_)
    plt.title('Target Image')
    plt.savefig(out_dir + '/target.png')

    input_scene = copy.deepcopy(scene)
    del input_scene['objects']['sphere']
    del input_scene['objects']['triangle']

    num_splats = width * height
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    z = tch_var_f(2 * np.random.rand(num_splats) - 1)
    z.requires_grad = True
    pos = torch.stack((tch_var_f(x.ravel()), tch_var_f(y.ravel()), z), dim=1)
    normals = tch_var_f(np.ones((num_splats, 4)) * np.array([0, 0, 1, 0]))
    normals.requires_grad = True
    material_idx = tch_var_l(np.ones(num_splats) * 3)

    input_scene['objects'] = {'disk': {'pos': pos,
                                       'normal': normals,
                                       'material_idx': material_idx
                                       }
                              }
    optimizer = optim.Adam((z, normals), lr=lr)

    h0 = plt.figure()
    h1 = plt.figure()
    loss_per_iter = []
    for iter in range(max_iter):
        res = render_splats_NDC(input_scene)
        im_out = res['image']

        optimizer.zero_grad()
        loss = criterion(scale * im_out, scale * target_im)

        im_out_ = get_data(im_out)
        loss_ = get_data(loss)
        loss_per_iter.append(loss_)

        if iter == 0:
            plt.figure(h0.number)
            plt.imshow(im_out_)
            plt.title('Initial')

        if iter % print_interval == 0 or iter == max_iter - 1:
            print('%d. loss= %f' % (iter, loss_))

        if iter % imsave_interval == 0 or iter == max_iter - 1:
            plt.figure(h1.number)
            plt.imshow(im_out_)
            plt.title('%d. loss= %f' % (iter, loss_))
            plt.savefig(out_dir + '/fig_%05d.png' % iter)

        loss.backward()
        optimizer.step()

    plt.figure()
    plt.plot(loss_per_iter, linewidth=2)
    plt.xlabel('Iteration', fontsize=14)
    plt.title('Loss', fontsize=12)
    plt.grid(True)
    plt.savefig(out_dir + '/loss.png')

    plt.ioff()
    plt.show()


def optimize_splats_along_ray_shadow_test(out_dir, width, height, max_iter=100, lr=1e-3, scale=10,
                                          shadow=True, vis_only=False, samples=1, est_normals=False,
                                          print_interval=10, imsave_interval=10, xyz_save_interval=100):
    """A demo function to check if the differentiable renderer can optimize splats rendered along ray.
    :param scene:
    :param out_dir:
    :return:
    """
    import torch
    import copy
    from diffrend.torch.params import SCENE_SPHERE_HALFBOX_0

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    scene = SCENE_SPHERE_HALFBOX_0
    scene['camera']['viewport'] = [0, 0, width, height]
    scene['camera']['fovy'] = np.deg2rad(45)
    scene['camera']['focal_length'] = 1
    scene['camera']['eye'] = tch_var_f([2, 1, 2, 1])  # tch_var_f([1, 1, 1, 1]) # tch_var_f([2, 2, 2, 1]) #
    scene['camera']['at'] = tch_var_f([0, 0.8, 0, 1])  # tch_var_f([0, 1, 0, 1]) # tch_var_f([2, 2, 0, 1])  #

    target_res = render(scene, tiled=True, shadow=shadow)
    target_im = normalize_maxmin(target_res['image'])
    target_im.require_grad = False
    target_im_ = get_data(target_im)
    target_pos_ = get_data(target_res['pos'])
    target_normal_ = get_data(target_res['normal'])
    target_normalmap_img_ = get_normalmap_image(target_normal_)
    target_depth_ = get_data(target_res['depth'])
    print('[z_min, z_max] = [%f, %f]' % (np.min(target_pos_[..., 2]), np.max(target_pos_[..., 2])))
    print('[depth_min, depth_max] = [%f, %f]' % (np.min(target_depth_), np.max(target_depth_)))

    # world -> cam -> render_splats_along_ray
    cc_tform = world_to_cam(target_res['pos'].view((-1, 3)), target_res['normal'].view((-1, 3)), scene['camera'])
    wc_cc_tform = cam_to_world(cc_tform['pos'], cc_tform['normal'], scene['camera'])

    pos_diff = torch.abs(wc_cc_tform['pos'][:, :3] - target_res['pos'].view((-1, 3)))
    mean_pos_diff = torch.mean(pos_diff)
    normal_diff = torch.abs(wc_cc_tform['normal'][:, :3] - target_res['normal'].view(-1, 3))
    mean_normal_diff = torch.mean(normal_diff)
    print('mean_pos_diff', mean_pos_diff, 'mean_normal_diff', mean_normal_diff)

    wc_cc_normal = wc_cc_tform['normal'].view(target_im_.shape)
    wc_cc_normal_img = get_normalmap_image(get_data(wc_cc_normal))

    material_idx = tch_var_l(np.ones(cc_tform['pos'].shape[0]) * 3)
    input_scene = copy.deepcopy(scene)
    del input_scene['objects']['sphere']
    del input_scene['objects']['triangle']
    light_vis = tch_var_f(np.ones((input_scene['lights']['pos'].shape[0], cc_tform['pos'].shape[0])))
    input_scene['objects'] = {'disk': {'pos': cc_tform['pos'],
                                       'normal': cc_tform['normal'],
                                       'material_idx': material_idx,
                                       'light_vis': light_vis,
                                       }
                              }
    target_res_noshadow = render(scene, tiled=True, shadow=False)
    res = render_splats_along_ray(input_scene)
    test_img_ = get_data(normalize_maxmin(res['image']))
    test_depth_ = get_data(res['depth'])
    test_normal_ = get_data(res['normal']).reshape(test_img_.shape)
    test_normalmap_ = get_normalmap_image(test_normal_)
    im_diff = np.abs(test_img_ - get_data(normalize_maxmin(target_res_noshadow['image'])))
    print('mean image diff: {}'.format(np.mean(im_diff)))
    #### PLOT
    plt.ion()
    plt.figure()
    plt.imshow(test_img_, interpolation='none')
    plt.title('Test Image')
    plt.savefig(out_dir + '/test_img.png')
    plt.figure()
    plt.imshow(test_depth_, interpolation='none')
    plt.title('Test Depth')
    plt.savefig(out_dir + '/test_depth.png')

    plt.figure()
    plt.imshow(test_normalmap_, interpolation='none')
    plt.title('Test Normals')
    plt.savefig(out_dir + '/test_normal.png')

    ####
    criterion = nn.L1Loss() #nn.MSELoss()
    criterion = criterion.cuda()

    plt.ion()
    plt.figure()
    plt.imshow(target_im_, interpolation='none')
    plt.title('Target Image')
    plt.savefig(out_dir + '/target.png')

    plt.figure()
    plt.imshow(target_normalmap_img_, interpolation='none')
    plt.title('Normals')
    plt.savefig(out_dir + '/normal.png')

    plt.figure()
    plt.imshow(wc_cc_normal_img, interpolation='none')
    plt.title('WC_CC Normals')
    plt.savefig(out_dir + '/wc_cc_normal.png')

    input_scene = copy.deepcopy(scene)
    del input_scene['objects']['sphere']
    del input_scene['objects']['triangle']
    input_scene['camera']['viewport'] = [0, 0, int(width / samples), int(height / samples)]

    num_splats = int(width * height / (samples * samples))
    x, y = np.meshgrid(np.linspace(-1, 1, int(width / samples)), np.linspace(-1, 1, int(height / samples)))
    z_min = scene['camera']['focal_length']
    z_max = 3

    z = -torch.clamp(tch_var_f(2 * np.random.rand(num_splats)), z_min, z_max)
    z.requires_grad = True

    normal_angles = tch_var_f(np.random.rand(num_splats, 2))
    normal_angles.requires_grad = True
    material_idx = tch_var_l(np.ones(num_splats) * 3)

    light_vis = tch_var_f(np.ones((input_scene['lights']['pos'].shape[0], num_splats)))
    light_vis.requires_grad = True

    if vis_only:
        assert shadow is True
        opt_vars = [light_vis]
        z = cc_tform['pos'][:, 2]
        # FIXME: sph2cart
        #normals = cc_tform['normal']
    else:
        opt_vars = [z, normal_angles]
        if shadow:
            opt_vars += [light_vis]

    optimizer = optim.Adam(opt_vars, lr=lr)
    lr_scheduler = StepLR(optimizer, step_size=10000, gamma=0.8)

    h0 = plt.figure()
    h1 = plt.figure()
    h2 = plt.figure()
    h3 = plt.figure()
    h4 = plt.figure()

    gs1 = gridspec.GridSpec(3, 3)
    gs1.update(wspace=0.0025, hspace=0.02)

    # Two options for z_norm_consistency
    # 1. start after N iterations
    # 2. start at the beginning and decay
    # 3. start after N iterations and decay to 0
    no_decay = lambda x: x
    exp_decay = lambda x, scale: torch.exp(-x / scale)
    linear_decay = lambda x, scale: scale / (x + 1e-6)

    spatial_var_loss_weight = 0.0
    normal_away_from_cam_loss_weight = 0.0
    spatial_loss_weight = 0.0

    z_norm_weight_init = 1e-2  # 1e-5
    z_norm_activate_iter = 0  # 1000
    decay_fn = lambda x: linear_decay(x, 100)
    loss_per_iter = []
    for iter in range(max_iter):
        lr_scheduler.step()
        phi = F.sigmoid(normal_angles[:, 0]) * 2 * np.pi
        theta = F.sigmoid(normal_angles[:, 1]) * np.pi / 2  # F.tanh(normal_angles[:, 1]) * np.pi / 2
        normals = sph2cart_unit(torch.stack((phi, theta), dim=1))

        pos = torch.stack((tch_var_f(x.ravel()), tch_var_f(y.ravel()), z), dim=1)

        input_scene['objects'] = {'disk': {'pos': pos,
                                           'normal': normalize(normals) if not est_normals else None,
                                           'material_idx': material_idx,
                                           'light_vis': torch.sigmoid(light_vis),
                                           }
                                  }
        res = render_splats_along_ray(input_scene, samples=samples)
        res_pos = res['pos']
        res_normal = res['normal']
        spatial_loss = spatial_3x3(res_pos)
        unit_normal_loss = unit_norm2_L2loss(normals, 10.0)
        normal_away_from_cam_loss = away_from_camera_penalty(res_pos, res_normal)
        z_pos = res_pos[..., 2]
        z_loss = torch.mean((10 * F.relu(z_min - torch.abs(z_pos))) ** 2 + (10 * F.relu(torch.abs(z_pos) - z_max)) ** 2)
        z_norm_loss = normal_consistency_cost(res_pos, res_normal, norm=1)
        spatial_var = torch.mean(res_pos[..., 0].var() + res_pos[..., 1].var() + res_pos[..., 2].var())
        spatial_var_loss = (1 / (spatial_var + 1e-4))
        im_out = normalize_maxmin(res['image'])
        res_depth_ = get_data(res['depth'])

        optimizer.zero_grad()
        z_norm_weight = z_norm_weight_init * float(iter > z_norm_activate_iter) * decay_fn(iter - z_norm_activate_iter)
        loss = criterion(scale * im_out, scale * target_im) + z_loss + unit_normal_loss + \
            z_norm_weight * z_norm_loss + \
            spatial_var_loss_weight * spatial_var_loss + \
            normal_away_from_cam_loss_weight * normal_away_from_cam_loss + \
            spatial_loss_weight * spatial_loss

        im_out_ = get_data(im_out)
        im_out_normal_ = get_data(res['normal'])
        pos_out_ = get_data(res['pos'])

        loss_ = get_data(loss)
        z_loss_ = get_data(z_loss)
        z_norm_loss_ = get_data(z_norm_loss)
        spatial_loss_ = get_data(spatial_loss)
        spatial_var_loss_ = get_data(spatial_var_loss)
        unit_normal_loss_ = get_data(unit_normal_loss)
        normal_away_from_cam_loss_ = get_data(normal_away_from_cam_loss)
        normals_ = get_data(res_normal)
        loss_per_iter.append(loss_)

        if iter == 0:
            plt.figure(h0.number)
            plt.imshow(im_out_)
            plt.title('Initial')

        if iter % print_interval == 0 or iter == max_iter - 1:
            z_ = get_data(z)
            z__ = pos_out_[..., 2]
            print('%d. loss= %f nloss=%f z_loss=%f [%f, %f] [%f, %f], z_normal_loss: %f,'
                  ' spatial_var_loss: %f, normal_away_loss: %f'
                  ' nz_range: [%f, %f], spatial_loss: %f' %
                  (iter, loss_, unit_normal_loss_, z_loss_, np.min(z_), np.max(z_), np.min(z__),
                   np.max(z__), z_norm_loss_, spatial_var_loss_, normal_away_from_cam_loss_,
                   normals_[..., 2].min(), normals_[..., 2].max(), spatial_loss_))

        if iter % xyz_save_interval == 0 or iter == max_iter - 1:
            save_xyz(out_dir + '/res_{:05d}.xyz'.format(iter), get_data(res_pos), get_data(res_normal))

        if iter % imsave_interval == 0 or iter == max_iter - 1:
            z_ = get_data(z)
            plt.figure(h4.number)
            plt.clf()
            plt.suptitle('%d. loss= %f [%f, %f]' % (iter, loss_, np.min(z_), np.max(z_)))
            plt.subplot(121)
            #plt.axis('off')
            plt.imshow(im_out_, interpolation='none')
            plt.title('Output')
            plt.subplot(122)
            #plt.axis('off')
            plt.imshow(target_im_, interpolation='none')
            plt.title('Ground truth')
            # plt.subplot(223)
            # plt.plot(loss_per_iter, linewidth=2)
            # plt.xlabel('Iteration', fontsize=14)
            # plt.title('Loss', fontsize=12)
            # plt.grid(True)
            plt.savefig(out_dir + '/fig_im_gt_loss_%05d.png' % iter)

            plt.figure(h1.number, figsize=(4, 4))
            plt.clf()
            plt.suptitle('%d. loss= %f [%f, %f]' % (iter, loss_, np.min(z_), np.max(z_)))
            plt.subplot(gs1[0])
            plt.axis('off')
            plt.imshow(im_out_, interpolation='none')
            plt.subplot(gs1[1])
            plt.axis('off')
            plt.imshow(get_normalmap_image(im_out_normal_), interpolation='none')
            ax = plt.subplot(gs1[2])
            plt.axis('off')
            im_tmp = ax.imshow(res_depth_, interpolation='none')
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im_tmp, cax=cax)


            plt.subplot(gs1[3])
            plt.axis('off')
            plt.imshow(target_im_, interpolation='none')
            plt.subplot(gs1[4])
            plt.axis('off')
            plt.imshow(test_normalmap_, interpolation='none')
            ax = plt.subplot(gs1[5])
            plt.axis('off')
            im_tmp = ax.imshow(test_depth_, interpolation='none')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im_tmp, cax=cax)

            W, H = input_scene['camera']['viewport'][2:]
            light_vis_ = get_data(torch.sigmoid(light_vis))
            plt.subplot(gs1[6])
            plt.axis('off')
            plt.imshow(light_vis_[0].reshape((H, W)), interpolation='none')

            if(light_vis_.shape[0] > 1):
                plt.subplot(gs1[7])
                plt.axis('off')
                plt.imshow(light_vis_[1].reshape((H, W)), interpolation='none')

            if (light_vis_.shape[0] > 2):
                plt.subplot(gs1[8])
                plt.axis('off')
                plt.imshow(light_vis_[2].reshape((H, W)), interpolation='none')


            plt.savefig(out_dir + '/fig_%05d.png' % iter)

            plt.figure(h2.number)
            plt.clf()
            plt.imshow(res_depth_)
            plt.colorbar()
            plt.savefig(out_dir + '/fig_depth_%05d.png' % iter)

            plt.figure(h3.number)
            plt.clf()
            plt.imshow(z_.reshape(H, W))
            plt.colorbar()
            plt.savefig(out_dir + '/fig_z_%05d.png' % iter)

        loss.backward()
        optimizer.step()

    plt.figure()
    plt.plot(loss_per_iter, linewidth=2)
    plt.xlabel('Iteration', fontsize=14)
    plt.title('Loss', fontsize=12)
    plt.grid(True)
    plt.savefig(out_dir + '/loss.png')

    plt.ioff()
    plt.show()


def generate_normals(z, camera, norm_est_net_fn):
    W, H = camera['viewport'][2:]
    pcl = z_to_pcl_CC(z.squeeze(), camera)
    n = norm_est_net_fn(pcl.view(H, W, 3).permute(2, 0, 1)[np.newaxis, ...])
    return n.squeeze().permute(1, 2, 0).view(-1, 3).contiguous()


def optimize_splats_along_ray_shadow_with_normalest_test(out_dir, width, height, max_iter=100, lr=1e-3, scale=10,
                                                         shadow=True, vis_only=False, samples=1, est_normals=False,
                                                         b_generate_normals=True, print_interval=10,
                                                         imsave_interval=10, xyz_save_interval=100):
    """A demo function to check if the differentiable renderer can optimize splats rendered along ray.
    :param scene:
    :param out_dir:
    :return:
    """
    import torch
    import copy
    from diffrend.torch.params import SCENE_SPHERE_HALFBOX_0

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    scene = SCENE_SPHERE_HALFBOX_0
    scene['camera']['viewport'] = [0, 0, width, height]
    scene['camera']['fovy'] = np.deg2rad(45)
    scene['camera']['focal_length'] = 1
    scene['camera']['eye'] = tch_var_f([2, 1, 2, 1])  # tch_var_f([1, 1, 1, 1]) # tch_var_f([2, 2, 2, 1]) #
    scene['camera']['at'] = tch_var_f([0, 0.8, 0, 1])  # tch_var_f([0, 1, 0, 1]) # tch_var_f([2, 2, 0, 1])  #

    target_res = render(scene, tiled=True, shadow=shadow)
    target_im = normalize_maxmin(target_res['image'])
    target_im.require_grad = False
    target_im_ = get_data(target_im)
    target_pos_ = get_data(target_res['pos'])
    target_normal_ = get_data(target_res['normal'])
    target_normalmap_img_ = get_normalmap_image(target_normal_)
    target_depth_ = get_data(target_res['depth'])
    print('[z_min, z_max] = [%f, %f]' % (np.min(target_pos_[..., 2]), np.max(target_pos_[..., 2])))
    print('[depth_min, depth_max] = [%f, %f]' % (np.min(target_depth_), np.max(target_depth_)))

    # world -> cam -> render_splats_along_ray
    cc_tform = world_to_cam(target_res['pos'].view((-1, 3)), target_res['normal'].view((-1, 3)), scene['camera'])
    wc_cc_tform = cam_to_world(cc_tform['pos'], cc_tform['normal'], scene['camera'])

    pos_diff = torch.abs(wc_cc_tform['pos'][:, :3] - target_res['pos'].view((-1, 3)))
    mean_pos_diff = torch.mean(pos_diff)
    normal_diff = torch.abs(wc_cc_tform['normal'][:, :3] - target_res['normal'].view(-1, 3))
    mean_normal_diff = torch.mean(normal_diff)
    print('mean_pos_diff', mean_pos_diff, 'mean_normal_diff', mean_normal_diff)

    wc_cc_normal = wc_cc_tform['normal'].view(target_im_.shape)
    wc_cc_normal_img = get_normalmap_image(get_data(wc_cc_normal))

    material_idx = tch_var_l(np.ones(cc_tform['pos'].shape[0]) * 3)
    input_scene = copy.deepcopy(scene)
    del input_scene['objects']['sphere']
    del input_scene['objects']['triangle']
    light_vis = tch_var_f(np.ones((input_scene['lights']['pos'].shape[0], cc_tform['pos'].shape[0])))
    input_scene['objects'] = {'disk': {'pos': cc_tform['pos'],
                                       'normal': cc_tform['normal'],
                                       'material_idx': material_idx,
                                       'light_vis': light_vis,
                                       }
                              }
    target_res_noshadow = render(scene, tiled=True, shadow=False)
    res = render_splats_along_ray(input_scene)
    test_img_ = get_data(normalize_maxmin(res['image']))
    test_depth_ = get_data(res['depth'])
    test_normal_ = get_data(res['normal']).reshape(test_img_.shape)
    test_normalmap_ = get_normalmap_image(test_normal_)
    im_diff = np.abs(test_img_ - get_data(normalize_maxmin(target_res_noshadow['image'])))
    print('mean image diff: {}'.format(np.mean(im_diff)))
    #### PLOT
    plt.ion()
    plt.figure()
    plt.imshow(test_img_, interpolation='none')
    plt.title('Test Image')
    plt.savefig(out_dir + '/test_img.png')
    plt.figure()
    plt.imshow(test_depth_, interpolation='none')
    plt.title('Test Depth')
    plt.savefig(out_dir + '/test_depth.png')

    plt.figure()
    plt.imshow(test_normalmap_, interpolation='none')
    plt.title('Test Normals')
    plt.savefig(out_dir + '/test_normal.png')

    ####
    criterion = nn.L1Loss() #nn.MSELoss()
    criterion = criterion.cuda()

    plt.ion()
    plt.figure()
    plt.imshow(target_im_, interpolation='none')
    plt.title('Target Image')
    plt.savefig(out_dir + '/target.png')

    plt.figure()
    plt.imshow(target_normalmap_img_, interpolation='none')
    plt.title('Normals')
    plt.savefig(out_dir + '/normal.png')

    plt.figure()
    plt.imshow(wc_cc_normal_img, interpolation='none')
    plt.title('WC_CC Normals')
    plt.savefig(out_dir + '/wc_cc_normal.png')

    input_scene = copy.deepcopy(scene)
    del input_scene['objects']['sphere']
    del input_scene['objects']['triangle']
    input_scene['camera']['viewport'] = [0, 0, int(width / samples), int(height / samples)]

    num_splats = int(width * height / (samples * samples))
    x, y = np.meshgrid(np.linspace(-1, 1, int(width / samples)), np.linspace(-1, 1, int(height / samples)))
    z_min = scene['camera']['focal_length']
    z_max = 3

    z = -torch.clamp(tch_var_f(2 * np.random.rand(num_splats)), z_min, z_max)
    z.requires_grad = True

    normal_angles = tch_var_f(np.random.rand(num_splats, 2))
    normal_angles.requires_grad = True
    material_idx = tch_var_l(np.ones(num_splats) * 3)

    light_vis = tch_var_f(np.ones((input_scene['lights']['pos'].shape[0], num_splats)))
    light_vis.requires_grad = True

    if vis_only:
        assert shadow is True
        opt_vars = [light_vis]
        z = cc_tform['pos'][:, 2]
        # FIXME: sph2cart
        #normals = cc_tform['normal']
    else:
        opt_vars = [z, normal_angles]
        if shadow:
            opt_vars += [light_vis]

    optimizer = optim.Adam(opt_vars, lr=lr)
    lr_scheduler = StepLR(optimizer, step_size=10000, gamma=0.8)

    h0 = plt.figure()
    h1 = plt.figure()
    h2 = plt.figure()
    h3 = plt.figure()
    h4 = plt.figure()

    gs1 = gridspec.GridSpec(3, 3)
    gs1.update(wspace=0.0025, hspace=0.02)

    # Two options for z_norm_consistency
    # 1. start after N iterations
    # 2. start at the beginning and decay
    # 3. start after N iterations and decay to 0
    no_decay = lambda x: x
    exp_decay = lambda x, scale: torch.exp(-x / scale)
    linear_decay = lambda x, scale: scale / (x + 1e-6)

    spatial_var_loss_weight = 0.0
    normal_away_from_cam_loss_weight = 0.0
    spatial_loss_weight = 0.0

    z_norm_weight_init = 1e-2  # 1e-5
    z_norm_activate_iter = 0  # 1000
    decay_fn = lambda x: linear_decay(x, 100)
    loss_per_iter = []
    normal_est_network = NEstNet(sph=True)
    for iter in range(max_iter):
        lr_scheduler.step()
        if b_generate_normals:
            normals = generate_normals(z, scene['camera'], normal_est_network)
        else:
            phi = F.sigmoid(normal_angles[:, 0]) * 2 * np.pi
            theta = F.sigmoid(normal_angles[:, 1]) * np.pi / 2  # F.tanh(normal_angles[:, 1]) * np.pi / 2
            normals = sph2cart_unit(torch.stack((phi, theta), dim=1))

        pos = torch.stack((tch_var_f(x.ravel()), tch_var_f(y.ravel()), z), dim=1)

        input_scene['objects'] = {'disk': {'pos': pos,
                                           'normal': normalize(normals) if not est_normals else None,
                                           'material_idx': material_idx,
                                           'light_vis': torch.sigmoid(light_vis),
                                           }
                                  }
        res = render_splats_along_ray(input_scene, samples=samples)
        res_pos = res['pos']
        res_normal = res['normal']
        spatial_loss = spatial_3x3(res_pos)
        unit_normal_loss = unit_norm2_L2loss(normals, 10.0)
        normal_away_from_cam_loss = away_from_camera_penalty(res_pos, res_normal)
        z_pos = res_pos[..., 2]
        z_loss = torch.mean((10 * F.relu(z_min - torch.abs(z_pos))) ** 2 + (10 * F.relu(torch.abs(z_pos) - z_max)) ** 2)
        z_norm_loss = normal_consistency_cost(res_pos, res_normal, norm=1)
        spatial_var = torch.mean(res_pos[..., 0].var() + res_pos[..., 1].var() + res_pos[..., 2].var())
        spatial_var_loss = (1 / (spatial_var + 1e-4))
        im_out = normalize_maxmin(res['image'])
        res_depth_ = get_data(res['depth'])

        optimizer.zero_grad()
        z_norm_weight = z_norm_weight_init * float(iter > z_norm_activate_iter) * decay_fn(iter - z_norm_activate_iter)
        loss = criterion(scale * im_out, scale * target_im) + z_loss + unit_normal_loss + \
            z_norm_weight * z_norm_loss + \
            spatial_var_loss_weight * spatial_var_loss + \
            normal_away_from_cam_loss_weight * normal_away_from_cam_loss + \
            spatial_loss_weight * spatial_loss

        im_out_ = get_data(im_out)
        im_out_normal_ = get_data(res['normal'])
        pos_out_ = get_data(res['pos'])

        loss_ = get_data(loss)
        z_loss_ = get_data(z_loss)
        z_norm_loss_ = get_data(z_norm_loss)
        spatial_loss_ = get_data(spatial_loss)
        spatial_var_loss_ = get_data(spatial_var_loss)
        unit_normal_loss_ = get_data(unit_normal_loss)
        normal_away_from_cam_loss_ = get_data(normal_away_from_cam_loss)
        normals_ = get_data(res_normal)
        loss_per_iter.append(loss_)

        if iter == 0:
            plt.figure(h0.number)
            plt.imshow(im_out_)
            plt.title('Initial')

        if iter % print_interval == 0 or iter == max_iter - 1:
            z_ = get_data(z)
            z__ = pos_out_[..., 2]
            print('%d. loss= %f nloss=%f z_loss=%f [%f, %f] [%f, %f], z_normal_loss: %f,'
                  ' spatial_var_loss: %f, normal_away_loss: %f'
                  ' nz_range: [%f, %f], spatial_loss: %f' %
                  (iter, loss_, unit_normal_loss_, z_loss_, np.min(z_), np.max(z_), np.min(z__),
                   np.max(z__), z_norm_loss_, spatial_var_loss_, normal_away_from_cam_loss_,
                   normals_[..., 2].min(), normals_[..., 2].max(), spatial_loss_))

        if iter % xyz_save_interval == 0 or iter == max_iter - 1:
            save_xyz(out_dir + '/res_{:05d}.xyz'.format(iter), get_data(res_pos), get_data(res_normal))

        if iter % imsave_interval == 0 or iter == max_iter - 1:
            z_ = get_data(z)
            plt.figure(h4.number)
            plt.clf()
            plt.suptitle('%d. loss= %f [%f, %f]' % (iter, loss_, np.min(z_), np.max(z_)))
            plt.subplot(121)
            #plt.axis('off')
            plt.imshow(im_out_, interpolation='none')
            plt.title('Output')
            plt.subplot(122)
            #plt.axis('off')
            plt.imshow(target_im_, interpolation='none')
            plt.title('Ground truth')
            # plt.subplot(223)
            # plt.plot(loss_per_iter, linewidth=2)
            # plt.xlabel('Iteration', fontsize=14)
            # plt.title('Loss', fontsize=12)
            # plt.grid(True)
            plt.savefig(out_dir + '/fig_im_gt_loss_%05d.png' % iter)

            plt.figure(h1.number, figsize=(4, 4))
            plt.clf()
            plt.suptitle('%d. loss= %f [%f, %f]' % (iter, loss_, np.min(z_), np.max(z_)))
            plt.subplot(gs1[0])
            plt.axis('off')
            plt.imshow(im_out_, interpolation='none')
            plt.subplot(gs1[1])
            plt.axis('off')
            plt.imshow(get_normalmap_image(im_out_normal_), interpolation='none')
            ax = plt.subplot(gs1[2])
            plt.axis('off')
            im_tmp = ax.imshow(res_depth_, interpolation='none')
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im_tmp, cax=cax)


            plt.subplot(gs1[3])
            plt.axis('off')
            plt.imshow(target_im_, interpolation='none')
            plt.subplot(gs1[4])
            plt.axis('off')
            plt.imshow(test_normalmap_, interpolation='none')
            ax = plt.subplot(gs1[5])
            plt.axis('off')
            im_tmp = ax.imshow(test_depth_, interpolation='none')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im_tmp, cax=cax)

            W, H = input_scene['camera']['viewport'][2:]
            light_vis_ = get_data(torch.sigmoid(light_vis))
            plt.subplot(gs1[6])
            plt.axis('off')
            plt.imshow(light_vis_[0].reshape((H, W)), interpolation='none')

            if(light_vis_.shape[0] > 1):
                plt.subplot(gs1[7])
                plt.axis('off')
                plt.imshow(light_vis_[1].reshape((H, W)), interpolation='none')

            if (light_vis_.shape[0] > 2):
                plt.subplot(gs1[8])
                plt.axis('off')
                plt.imshow(light_vis_[2].reshape((H, W)), interpolation='none')


            plt.savefig(out_dir + '/fig_%05d.png' % iter)

            plt.figure(h2.number)
            plt.clf()
            plt.imshow(res_depth_)
            plt.colorbar()
            plt.savefig(out_dir + '/fig_depth_%05d.png' % iter)

            plt.figure(h3.number)
            plt.clf()
            plt.imshow(z_.reshape(H, W))
            plt.colorbar()
            plt.savefig(out_dir + '/fig_z_%05d.png' % iter)

        loss.backward()
        optimizer.step()

    plt.figure()
    plt.plot(loss_per_iter, linewidth=2)
    plt.xlabel('Iteration', fontsize=14)
    plt.title('Loss', fontsize=12)
    plt.grid(True)
    plt.savefig(out_dir + '/loss.png')

    plt.ioff()
    plt.show()


def test_scalability(filename, out_dir='./test_scale'):
    # GTX 980 8GB
    # 320 x 240 250 objs
    # 64 x 64 5000 objs
    # 32 x 32 20000 objs
    # 16 x 16 75000 objs (slow)
    from diffrend.model import load_model

    splats = load_model(filename)
    v = splats['v']
    # normalize the vertices
    v = (v - np.mean(v, axis=0)) / (v.max() - v.min())

    print(np.min(splats['v'], axis=0))
    print(np.max(splats['v'], axis=0))
    print(np.min(v, axis=0))
    print(np.max(v, axis=0))

    rand_idx = np.arange(v.shape[0]) #np.random.randint(0, splats['v'].shape[0], 4000)  #
    large_scene = copy.deepcopy(SCENE_BASIC)

    large_scene['camera']['viewport'] = [0, 0, 64, 64] #[0, 0, 320, 240]
    large_scene['camera']['fovy'] = np.deg2rad(5.)
    large_scene['camera']['focal_length'] = 2.
    #large_scene['camera']['eye'] = tch_var_f([0.0, 1.0, 5.0, 1.0]),
    large_scene['objects']['disk']['pos'] = tch_var_f(v[rand_idx])
    large_scene['objects']['disk']['normal'] = tch_var_f(splats['vn'][rand_idx])
    large_scene['objects']['disk']['radius'] = tch_var_f(splats['r'][rand_idx].ravel() * 2)
    large_scene['objects']['disk']['material_idx'] = tch_var_l(np.zeros(rand_idx.size, dtype=int).tolist())
    large_scene['materials']['albedo'] = tch_var_f([[0.6, 0.6, 0.6]])

    render_scene(large_scene, out_dir, plot_res=True)


if __name__ == '__main__':
    import copy
    from data import DIR_DATA
    parser = argparse.ArgumentParser(usage='python main.py --[render|opt|test_scale]')
    parser.add_argument('--render', action='store_true', help='Renders a scene if specified')
    parser.add_argument('--opt', action='store_true', help='Optimizes material parameters if specified')
    parser.add_argument('--test_scale', action='store_true', help='Test what is the maximum number of splats that can'
                                                                  'be rendered at a given resolution.')
    parser.add_argument('--norm_depth_image_only', action='store_true', default=False,
                        help='Only render the normalized depth image.')
    parser.add_argument('--backface_culling', action='store_true', help='Filter out objects that are facing away from'
                                                                        'the camera.')
    parser.add_argument('--out_dir', type=str, default='./output')
    parser.add_argument('--model_filename', type=str, default=DIR_DATA + '/bunny.splat',
                        help='Input model filename needed for scalability testing')
    parser.add_argument('--display', action='store_true', help='Display result using matplotlib.')
    parser.add_argument('--ortho', action='store_true', help='Use Orthographic Projection.')
    parser.add_argument('--opt-ndc-test', action='store_true', help='Test optimization in NDC.')
    parser.add_argument('--opt-ray-test', action='store_true', help='Test optimization render along ray.')
    parser.add_argument('--opt-ray-shadow-test', action='store_true', help='Test optimization render along ray.')
    parser.add_argument('--vis-only', action='store_true', help='Optimize only shadow map.')
    parser.add_argument('--est-normals', action='store_true', help='Estimates normals from position if enabled.')
    parser.add_argument('--samples', type=int, default=1, help='Pixel supersampling.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimization.')
    parser.add_argument('--max-iter', type=int, default=2000, help='Maximum number of iterations.')
    parser.add_argument('--print-interval', type=int, default=100, help='Print interval for optimization.')
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--height', type=int, default=64)

    args = parser.parse_args()
    print(args)
    #if not (args.render or args.opt or args.test_scale):
    #    args.render = True

    scene = SCENE_1
    if args.ortho:
        scene['camera']['proj_type'] = 'ortho'
    if args.render:
        res = render_scene(scene, args.out_dir, args.norm_depth_image_only, backface_culling=args.backface_culling,
                           plot_res=args.display)
    if args.opt:
        input_scene = copy.deepcopy(SCENE_BASIC)
        input_scene['materials']['albedo'] = tch_var_f([
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.1, 0.8, 0.9],
            [0.1, 0.8, 0.9],
            [0.9, 0.1, 0.1],
        ])
        optimize_scene(input_scene, scene, args.out_dir, max_iter=args.max_iter, lr=args.lr,
                       print_interval=args.print_interval)
    if args.test_scale:
        test_scalability(filename=args.model_filename, out_dir=args.out_dir)

    if args.opt_ndc_test:
        optimize_NDC_test(out_dir=args.out_dir, width=args.width, height=args.height,
                          max_iter=args.max_iter, lr=args.lr, print_interval=args.print_interval)

    if args.opt_ray_shadow_test or args.opt_ray_test:
        optimize_splats_along_ray_shadow_with_normalest_test(out_dir=args.out_dir, width=args.width, height=args.height,
                                              max_iter=args.max_iter, lr=args.lr,
                                              vis_only=args.vis_only, shadow=args.opt_ray_shadow_test,
                                              samples=args.samples, est_normals=args.est_normals,
                                              print_interval=args.print_interval)
