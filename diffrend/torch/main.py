from diffrend.torch.params import SCENE_BASIC, SCENE_1, SCENE_2
from diffrend.torch.renderer import render, render_splats_NDC, render_splats_along_ray
from diffrend.torch.utils import tch_var_f, tch_var_l, CUDA, get_data
import torch.nn as nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
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


def optimize_splats_along_ray_test(out_dir, width=32, height=32, max_iter=100, lr=1e-3, scale=10, print_interval=10,
                                   imsave_interval=10):
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
    scene['camera']['eye'] = tch_var_f([2, 1, 2, 1])
    scene['camera']['at'] = tch_var_f([0, 0.8, 0, 1])

    target_res = render(SCENE_SPHERE_HALFBOX_0)
    target_im = target_res['image']
    target_im.require_grad = False
    target_im_ = get_data(target_res['image'])
    target_pos_ = get_data(target_res['pos'])
    target_normal_ = get_data(target_res['normal'])
    print('[z_min, z_max] = [%f, %f]' % (np.min(target_pos_[..., 2]), np.max(target_pos_[..., 2])))
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
    z = -tch_var_f(2 * np.random.rand(num_splats))
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
    h2 = plt.figure()
    h3 = plt.figure()
    loss_per_iter = []
    for iter in range(max_iter):
        res = render_splats_along_ray(input_scene, use_old_sign=True)
        im_out = res['image']
        res_depth_ = get_data(res['depth'])

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
            z_ = get_data(z)
            print('%d. loss= %f [%f, %f]' % (iter, loss_, np.min(z_), np.max(z_)))

        if iter % imsave_interval == 0 or iter == max_iter - 1:
            z_ = get_data(z)
            plt.figure(h1.number)
            plt.imshow(im_out_)
            plt.title('%d. loss= %f [%f, %f]' % (iter, loss_, np.min(z_), np.max(z_)))
            plt.savefig(out_dir + '/fig_%05d.png' % iter)

            plt.figure(h2.number)
            plt.clf()
            plt.imshow(res_depth_)
            plt.colorbar()
            plt.savefig(out_dir + '/fig_depth_%05d.png' % iter)

            plt.figure(h3.number)
            plt.clf()
            plt.imshow(z_.reshape((32, 32)))
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
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimization.')
    parser.add_argument('--max-iter', type=int, default=2000, help='Maximum number of iterations.')
    parser.add_argument('--print-interval', type=int, default=100, help='Print interval for optimization.')
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--height', type=int, default=32)

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
    if args.opt_ray_test:
        optimize_splats_along_ray_test(out_dir=args.out_dir, width=args.width, height=args.height,
                                       max_iter=args.max_iter, lr=args.lr, print_interval=args.print_interval)
