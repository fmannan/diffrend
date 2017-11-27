from diffrend.torch.params import OUTPUT_FOLDER, SCENE_BASIC
from diffrend.torch.renderer import render
from diffrend.torch.utils import tch_var_f, tch_var_l, CUDA
import torch.nn as nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def render_scene(scene, output_folder, plot_res=True):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # main render run
    res = render(scene)
    if CUDA:
        im = res['image'].cpu().data.numpy()
    else:
        im = res['image'].data.numpy()

    if plot_res:
        plt.ion()
        plt.figure()
        plt.imshow(im)
        plt.title('Final Rendered Image')
        plt.savefig(output_folder + '/img_torch.png')

    if CUDA:
        depth = res['depth'].cpu().data.numpy()
    else:
        depth = res['depth'].data.numpy()
    depth[depth >= scene['camera']['far']] = np.inf
    print(depth.min(), depth.max())
    if plot_res:
        plt.figure()
        plt.imshow(depth)
        plt.title('Depth Image')
        plt.savefig(output_folder + '/img_depth_torch.png')

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

        if CUDA:
            im_out_ = im_out.cpu().data.numpy()
            loss_ = loss.cpu().data.numpy()
        else:
            im_out_ = im_out.data.numpy()
            loss_ = loss.data.numpy()

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
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--opt', action='store_true')
    parser.add_argument('--test_scale', action='store_true')
    parser.add_argument('--out_dir', type=str, default=OUTPUT_FOLDER)
    parser.add_argument('--model_filename', type=str, default=DIR_DATA + '/bunny.splat',
                        help='Input model filename needed for scalability testing')

    args = parser.parse_args()
    print(args)
    if not (args.render or args.opt or args.test_scale):
        args.render = True

    scene = SCENE_BASIC
    if args.render:
        res = render_scene(scene, args.out_dir)
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
        optimize_scene(input_scene, scene, args.out_dir, max_iter=2000, lr=1e-3, print_interval=100)
    if args.test_scale:
        test_scalability(filename=args.model_filename, out_dir=args.out_dir)
