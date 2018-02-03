from diffrend.torch.params import SCENE_BASIC
from diffrend.torch.utils import tch_var_f, tch_var_l, get_data
from diffrend.torch.renderer import render
from diffrend.numpy.ops import normalize as np_normalize
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
                               fovy, focal_length, norm_depth_image_only, cam_pos=None, b_display=False):
    """
    Randomly generate N samples on a surface and render them. The samples include position and normal, the radius is set
    to a constant.
    """
    sampling_time = []
    rendering_time = []

    obj = load_model(filename)
    # normalize the vertices
    v = obj['v']
    axis_range = np.max(v, axis=0) - np.min(v, axis=0)
    v = (v - np.mean(v, axis=0)) / max(axis_range)  # Normalize to make the largest spread 1
    obj['v'] = v

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
    if cam_pos is None:
        cam_pos = uniform_sample_sphere(radius=cam_dist, num_samples=num_views)
    obj_center = np.mean(v, axis=0)
    large_scene['camera']['at'] = tch_var_f(obj_center)

    if b_display:
        plt.figure()
    for idx in range(cam_pos.shape[0]):
        start_time = time()
        v, vn = uniform_sample_mesh(obj, num_samples=num_samples)
        sampling_time.append(time() - start_time)

        large_scene['objects']['disk']['pos'] = tch_var_f(v)
        large_scene['objects']['disk']['normal'] = tch_var_f(vn)

        large_scene['camera']['eye'] = tch_var_f(cam_pos[idx])
        suffix = '_{}'.format(idx)

        # main render run
        start_time = time()
        res = render(large_scene, norm_depth_image_only=norm_depth_image_only)
        rendering_time.append(time() - start_time)

        im = get_data(res['image'])
        depth = get_data(res['depth'])

        depth[depth >= large_scene['camera']['far']] = depth.min()
        im_depth = np.uint8(255. * (depth - depth.min()) / (depth.max() - depth.min()))

        if b_display:
            plt.imshow(im)
            plt.title('Image')
            plt.savefig(out_dir + '/fig_img' + suffix + '.png')

            plt.imshow(im_depth)
            plt.title('Depth Image')
            plt.savefig(out_dir + '/fig_depth' + suffix + '.png')

        imsave(out_dir + '/img' + suffix + '.png', im)
        imsave(out_dir + '/depth' + suffix + '.png', im_depth)

    # Timing statistics
    print('Sampling time mean: {}s, std: {}s'.format(np.mean(sampling_time), np.std(sampling_time)))
    print('Rendering time mean: {}s, std: {}s'.format(np.mean(rendering_time), np.std(rendering_time)))


def render_sphere(out_dir, cam_pos, radius, width, height, fovy, focal_length, num_views, std_z=0.01, std_normal=0.01,
                  b_display=False):
    """
    Randomly generate N samples on a surface and render them. The samples include position and normal, the radius is set
    to a constant.
    """
    print('render sphere')
    sampling_time = []
    rendering_time = []

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

    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    #z = np.sqrt(1 - np.min(np.stack((x ** 2 + y ** 2, np.ones_like(x)), axis=-1), axis=-1))
    unit_disk_mask = (x ** 2 + y ** 2) <= 1
    z = np.sqrt(1 - unit_disk_mask * (x ** 2 + y ** 2))

    # Make a hemi-sphere bulging out of the xy-plane scene
    z[~unit_disk_mask] = 0
    pos = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)

    # Normals outside the sphere should be [0, 0, 1]
    x[~unit_disk_mask] = 0
    y[~unit_disk_mask] = 0
    z[~unit_disk_mask] = 1

    normals = np_normalize(np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1))

    if b_display:
        plt.ion()
        plt.figure()
        plt.imshow(pos[..., 2].reshape((height, width)))

        plt.figure()
        plt.imshow(normals[..., 2].reshape((height, width)))

    large_scene['objects']['disk']['pos'] = tch_var_f(pos)
    large_scene['objects']['disk']['normal'] = tch_var_f(normals)

    large_scene['camera']['eye'] = tch_var_f(cam_pos)

    # main render run
    start_time = time()
    res = render(large_scene)
    rendering_time.append(time() - start_time)

    im = get_data(res['image'])
    depth = get_data(res['depth'])

    depth[depth >= large_scene['camera']['far']] = depth.min()
    im_depth = np.uint8(255. * (depth - depth.min()) / (depth.max() - depth.min()))

    if b_display:
        plt.figure()
        plt.imshow(im)
        plt.title('Image')
        plt.savefig(out_dir + '/fig_img_orig.png')

        plt.figure()
        plt.imshow(im_depth)
        plt.title('Depth Image')
        plt.savefig(out_dir + '/fig_depth_orig.png')

    imsave(out_dir + '/img_orig.png', im)
    imsave(out_dir + '/depth_orig.png', im_depth)

    # generate noisy data
    if b_display:
        h1 = plt.figure()
        h2 = plt.figure()
    noisy_pos = pos
    for view_idx in range(num_views):
        noisy_pos[..., 2] = pos[..., 2] + std_z * np.random.randn(num_samples)
        noisy_normals = np_normalize(normals + std_normal * np.random.randn(num_samples, 3))

        large_scene['objects']['disk']['pos'] = tch_var_f(noisy_pos)
        large_scene['objects']['disk']['normal'] = tch_var_f(noisy_normals)

        large_scene['camera']['eye'] = tch_var_f(cam_pos)

        # main render run
        start_time = time()
        res = render(large_scene)
        rendering_time.append(time() - start_time)

        im = get_data(res['image'])
        depth = get_data(res['depth'])

        depth[depth >= large_scene['camera']['far']] = depth.min()
        im_depth = np.uint8(255. * (depth - depth.min()) / (depth.max() - depth.min()))

        suffix_str = '{:05d}'.format(view_idx)

        if b_display:
            plt.figure(h1.number)
            plt.imshow(im)
            plt.title('Image')
            plt.savefig(out_dir + '/fig_img_' + suffix_str + '.png')

            plt.figure(h2.number)
            plt.imshow(im_depth)
            plt.title('Depth Image')
            plt.savefig(out_dir + '/fig_depth_' + suffix_str + '.png')

        imsave(out_dir + '/img_' + suffix_str + '.png', im)
        imsave(out_dir + '/depth_' + suffix_str + '.png', im_depth)

    # hold matplotlib figure
    plt.ioff()
    plt.show()


def preset_cam_pos_0():
    return np.array([[3.4065832, -1.26789198, 4.77364021],
                     [1.73761297, -4.62854391, -3.39960033],
                     [-2.03811183, 1.50594331, 5.43858758],
                     [-1.14525852, 5.44444109, -2.24642919],
                     [-3.72136062, -4.70636702, 0.03980693],
                     [-2.16651127, 3.45342292, 4.40228339],
                     [-3.39866041, 1.95044609, -4.54366234],
                     [-1.80005058, -5.62148377, 1.07644697],
                     [2.74616603, -1.86863041, 4.99667815],
                     [-5.78391582, -1.46707216, 0.62770774],
                     [-1.56162523, -5.78793779, 0.24718981],
                     [-1.29602688, 4.76694896, -3.40536517],
                     [-2.49890985, -5.27582671, -1.38603826],
                     [4.6751337, -3.44721669, -1.50327042],
                     [0.76567273, 5.60232432, -2.00691493],
                     [0.55733763, -3.1450544, 5.07917391],
                     [3.61504468, 4.28476875, -2.13827237],
                     [-4.1440199, 4.3368818, 0.13621782],
                     [0.299033, -0.18254953, -5.98976251],
                     [-1.52430796, -4.79467686, 3.26918324]])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(usage="splat_gen_render_demo.py --model filename --out_dir output_dir "
                                           "--n 5000 --width 128 --height 128 --r 0.025 --cam_dist 5 --nv 10")
    parser.add_argument('--model', type=str, default=DIR_DATA + '/chair_0001.off', help='Path to the model file')
    parser.add_argument('--out_dir', type=str, default='./render_samples/', help='Directory for rendered images.')
    parser.add_argument('--width', type=int, default=128, help='Width of output image.')
    parser.add_argument('--height', type=int, default=128, help='Height of output image.')
    parser.add_argument('--n', type=int, default=5000, help='Number of samples to generate.')
    parser.add_argument('--r', type=float, default=0.025, help='Constant radius for each splat.')
    parser.add_argument('--cam_dist', type=float, default=5.0, help='Camera distance from the center of the object.')
    parser.add_argument('--nv', type=int, default=10, help='Number of views to generate.')
    parser.add_argument('--fovy', type=float, default=18.0, help='Field of view in the vertical direction.')
    parser.add_argument('--f', type=float, default=0.1, help='Focal length of camera.')
    parser.add_argument('--norm_depth_image_only', action='store_true', default=False, help='Render on the normalized'
                                                                                            ' depth image.')
    parser.add_argument('--test_cam_dist', action='store_true', help='Check if the images are consistent with a'
                                                                     'camera at a fixed distance.')
    parser.add_argument('--display', action='store_true', help='Optionally display using matplotlib.')
    parser.add_argument('--render-sphere', action='store_true', help='Only render a sphere.')

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.render_sphere:
        render_sphere(out_dir=args.out_dir, cam_pos=[0, 0, 10], radius=0.03, width=64, height=64,
                      fovy=11.5, focal_length=args.f, num_views=args.nv)
    else:
        cam_pos = None
        if args.test_cam_dist:
            cam_pos = preset_cam_pos_0()

        render_random_splat_camera(filename=args.model, out_dir=args.out_dir, radius=args.r, num_samples=args.n,
                                   cam_dist=args.cam_dist, num_views=args.nv,
                                   width=args.width, height=args.height,
                                   fovy=args.fovy, focal_length=args.f,
                                   norm_depth_image_only=args.norm_depth_image_only,
                                   cam_pos=cam_pos, b_display=args.display)
