from diffrend.torch.params import SCENE_BASIC
from diffrend.torch.utils import tch_var_f, tch_var_l, CUDA
from diffrend.torch.renderer import render
from diffrend.utils.sample_generator import uniform_sample_mesh, uniform_sample_sphere
from diffrend.model import load_model
from data import DIR_DATA

import copy
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave


def render_random_splat_camera(filename, out_dir, num_samples, radius, cam_dist, num_views, width, height,
                               fovy, focal_length):
    """
    Randomly generate N samples on a surface and render them. The samples include position and normal, the radius is set
    to a constant.
    """
    sampling_time = []
    rendering_time = []

    obj = load_model(filename)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

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
    cam_pos = uniform_sample_sphere(radius=cam_dist, num_samples=num_views)
    plt.figure()
    for idx in range(cam_pos.shape[0]):
        start_time = time()
        v, vn = uniform_sample_mesh(obj, num_samples=num_samples)
        sampling_time.append(time() - start_time)

        # normalize the vertices
        v = (v - np.mean(v, axis=0)) / (v.max() - v.min())

        large_scene['objects']['disk']['pos'] = tch_var_f(v)
        large_scene['objects']['disk']['normal'] = tch_var_f(vn)

        large_scene['camera']['eye'] = tch_var_f(cam_pos[idx])
        suffix = '_{}'.format(idx)

        # main render run
        start_time = time()
        res = render(large_scene)
        rendering_time.append(time() - start_time)

        if CUDA:
            im = res['image'].cpu().data.numpy()
        else:
            im = res['image'].data.numpy()

        if CUDA:
            depth = res['depth'].cpu().data.numpy()
        else:
            depth = res['depth'].data.numpy()

        depth[depth >= large_scene['camera']['far']] = depth.min()
        im_depth = np.uint8(255. * (depth - depth.min()) / (depth.max() - depth.min()))
        plt.imshow(im)
        plt.title('Image')
        plt.savefig(out_dir + '/fig_img' + suffix + '.png')

        plt.imshow(im_depth)
        plt.title('Depth Image')
        plt.savefig(out_dir + '/fig_img_depth' + suffix + '.png')

        imsave(out_dir + '/img' + suffix + '.png', im)
        imsave(out_dir + '/img_depth' + suffix + '.png', im_depth)

    # Timing statistics
    print('Sampling time mean: {}s, std: {}s'.format(np.mean(sampling_time), np.std(sampling_time)))
    print('Rendering time mean: {}s, std: {}s'.format(np.mean(rendering_time), np.std(rendering_time)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(usage="splat_gen_render_demo.py --model filename --out_dir output_dir "
                                           "--n 5000 --width 128 --height 128 --r 0.025 --cam_dist 5 --nv 10")
    parser.add_argument('--model', type=str, default=DIR_DATA + '/chair_0001.off')
    parser.add_argument('--out_dir', type=str, default='./render_samples/')
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--r', type=float, default=0.025)
    parser.add_argument('--cam_dist', type=float, default=6.0, help='Camera distance from the center of the object')
    parser.add_argument('--nv', type=int, default=10, help='Number of views to generate')
    parser.add_argument('--fovy', type=float, default=15.0, help='Field of view in the vertical direction')
    parser.add_argument('--f', type=float, default=0.1, help='focal length')

    args = parser.parse_args()
    print(args)

    render_random_splat_camera(filename=args.model, out_dir=args.out_dir, radius=args.r, num_samples=args.n,
                               cam_dist=args.cam_dist, num_views=args.nv,
                               width=args.width, height=args.height,
                               fovy=args.fovy, focal_length=args.f)



