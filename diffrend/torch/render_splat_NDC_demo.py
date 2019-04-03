import numpy as np
import torch
from diffrend.torch.utils import tch_var_f, tch_var_l, get_data, lookat, lookat_inv, cam_to_world
from diffrend.torch.renderer import render_splats_NDC, render, render_splats_along_ray
from diffrend.torch.ops import perspective, inv_perspective
from diffrend.numpy.ops import normalize as np_normalize
from imageio import imsave
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from time import time


SCENE_TEST = {
    'camera': {
        'proj_type': 'perspective',
        'viewport': [0, 0, 2, 2],
        'fovy': np.deg2rad(90.),
        'focal_length': 1.,
        'eye': tch_var_f([0.0, 1.0, 10.0, 1.0]),
        'up': tch_var_f([0.0, 1.0, 0.0, 0.0]),
        'at': tch_var_f([0.0, 0.0, 0.0, 1.0]),
        'near': 1.0,
        'far': 1000.0,
    },
    'lights': {
        'pos': tch_var_f([
            [0., 0., -10., 1.0],
            [-15, 3, 15, 1.0],
            [0, 0., 10., 1.0],
        ]),
        'color_idx': tch_var_l([2, 1, 3]),
        # Light attenuation factors have the form (kc, kl, kq) and eq: 1/(kc + kl * d + kq * d^2)
        'attenuation': tch_var_f([
            [1., 0., 0.0],
            [0., 0., 0.01],
            [0., 0., 0.01],
        ]),
        'ambient': tch_var_f([0.01, 0.01, 0.01])
    },
    'colors': tch_var_f([
        [0.0, 0.0, 0.0],
        [0.8, 0.1, 0.1],
        [0.0, 0.0, 0.8],
        [0.2, 0.8, 0.2],
    ]),
    'materials': {
        'albedo': tch_var_f([
            [0.6, 0.6, 0.6],
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.5, 0.5, 0.5],
            [0.9, 0.1, 0.1],
            [0.1, 0.6, 0.8],
        ]),
        'coeffs': tch_var_f([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.4, 8.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
    },
    'objects': {
        'disk': {
            'normal': tch_var_f([
                [0., 0., 1., 0.0],
                [0., 1.0, 0.0, 0.0],
                [-1., -1.0, 1., 0.0],
                [0., 0., -1., 0.0],
            ]),
            'pos': tch_var_f([
                [-0.5, -0.5, 0., 1.0],
                [0.5, -0.5, 0.5, 1.0],
                [-.5, 0.5, 0.8, 1.0],
                [.5, .5, .2, 1.0],
            ]),
            'radius': tch_var_f([4, 7, 4, 3]),
            'material_idx': tch_var_l([4, 3, 5, 3])
        },
    },
    'tonemap': {
        'type': 'gamma',
        'gamma': tch_var_f([0.8])
    },
}


def render_sphere_world(out_dir, cam_pos, radius, width, height, fovy, focal_length,
                        b_display=False):
    """
    Generate z positions on a grid fixed inside the view frustum in the world coordinate system. Place the camera and
    choose the camera's field of view so that the side of the square touches the frustum.
    """
    import copy
    print('render sphere')
    sampling_time = []
    rendering_time = []

    num_samples = width * height
    r = np.ones(num_samples) * radius

    large_scene = copy.deepcopy(SCENE_TEST)

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
    im = np.uint8(255. * im)

    depth = get_data(res['depth'])
    depth[depth >= large_scene['camera']['far']] = depth.min()
    im_depth = np.uint8(255. * (depth - depth.min()) / (depth.max() - depth.min()))

    if b_display:
        plt.figure()
        plt.imshow(im, interpolation='none')
        plt.title('Image')
        plt.savefig(out_dir + '/fig_img_orig.png')

        plt.figure()
        plt.imshow(im_depth, interpolation='none')
        plt.title('Depth Image')
        plt.savefig(out_dir + '/fig_depth_orig.png')

    imsave(out_dir + '/img_orig.png', im)
    imsave(out_dir + '/depth_orig.png', im_depth)

    # hold matplotlib figure
    plt.ioff()
    plt.show()


def test_sphere_splat_NDC(out_dir, cam_pos, width, height, fovy, focal_length,  b_display=False):
    """
    Create a sphere on a square as in render_sphere_world, and then convert to the camera's coordinate system and to
    NDC and then render using render_splat_NDC.
    """
    import copy
    print('render sphere')
    sampling_time = []
    rendering_time = []

    num_samples = width * height

    large_scene = copy.deepcopy(SCENE_TEST)

    large_scene['camera']['viewport'] = [0, 0, width, height]
    large_scene['camera']['eye'] = tch_var_f(cam_pos)
    large_scene['camera']['fovy'] = np.deg2rad(fovy)
    large_scene['camera']['focal_length'] = focal_length
    large_scene['objects']['disk']['material_idx'] = tch_var_l(np.zeros(num_samples, dtype=int).tolist())
    large_scene['materials']['albedo'] = tch_var_f([[0.6, 0.6, 0.6]])
    large_scene['tonemap']['gamma'] = tch_var_f([1.0])  # Linear output

    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    #z = np.sqrt(1 - np.min(np.stack((x ** 2 + y ** 2, np.ones_like(x)), axis=-1), axis=-1))
    unit_disk_mask = (x ** 2 + y ** 2) <= 1
    z = np.sqrt(1 - unit_disk_mask * (x ** 2 + y ** 2))

    # Make a hemi-sphere bulging out of the xy-plane scene
    z[~unit_disk_mask] = 0
    pos = np.stack((x.ravel(), y.ravel(), z.ravel(), np.ones(num_samples)), axis=1)

    # Normals outside the sphere should be [0, 0, 1]
    x[~unit_disk_mask] = 0
    y[~unit_disk_mask] = 0
    z[~unit_disk_mask] = 1

    normals = np_normalize(np.stack((x.ravel(), y.ravel(), z.ravel(), np.zeros(num_samples)), axis=1))

    if b_display:
        plt.ion()
        plt.figure()
        plt.imshow(pos[..., 2].reshape((height, width)))

        plt.figure()
        plt.imshow(normals[..., 2].reshape((height, width)))

    # Convert to the camera's coordinate system
    Mcam = lookat(eye=large_scene['camera']['eye'], at=large_scene['camera']['at'], up=large_scene['camera']['up'])
    Mproj = perspective(fovy=large_scene['camera']['fovy'], aspect=width/height, near=large_scene['camera']['near'],
                        far=large_scene['camera']['far'])

    pos_CC = torch.matmul(tch_var_f(pos), Mcam.transpose(1, 0))
    pos_NDC = torch.matmul(pos_CC, Mproj.transpose(1, 0))

    large_scene['objects']['disk']['pos'] = pos_NDC / pos_NDC[..., 3][:, np.newaxis]
    large_scene['objects']['disk']['normal'] = tch_var_f(normals)

    # main render run
    start_time = time()
    res = render_splats_NDC(large_scene)
    rendering_time.append(time() - start_time)

    im = get_data(res['image'])
    im = np.uint8(255. * im)

    depth = get_data(res['depth'])
    depth[depth >= large_scene['camera']['far']] = depth.min()
    im_depth = np.uint8(255. * (depth - depth.min()) / (depth.max() - depth.min()))

    if b_display:
        plt.figure()
        plt.imshow(im, interpolation='none')
        plt.title('Image')
        plt.savefig(out_dir + '/fig_img_orig.png')

        plt.figure()
        plt.imshow(im_depth, interpolation='none')
        plt.title('Depth Image')
        plt.savefig(out_dir + '/fig_depth_orig.png')

    imsave(out_dir + '/img_orig.png', im)
    imsave(out_dir + '/depth_orig.png', im_depth)

    # hold matplotlib figure
    plt.ioff()
    plt.show()


def test_sphere_splat_render_along_ray(out_dir, cam_pos, width, height, fovy, focal_length, use_quartic,
                                       b_display=False):
    """
    Create a sphere on a square as in render_sphere_world, and then convert to the camera's coordinate system
    and then render using render_splats_along_ray.
    """
    import copy
    print('render sphere along ray')
    sampling_time = []
    rendering_time = []

    num_samples = width * height

    large_scene = copy.deepcopy(SCENE_TEST)

    large_scene['camera']['viewport'] = [0, 0, width, height]
    large_scene['camera']['eye'] = tch_var_f(cam_pos)
    large_scene['camera']['fovy'] = np.deg2rad(fovy)
    large_scene['camera']['focal_length'] = focal_length
    large_scene['objects']['disk']['material_idx'] = tch_var_l(np.zeros(num_samples, dtype=int).tolist())
    large_scene['materials']['albedo'] = tch_var_f([[0.6, 0.6, 0.6]])
    large_scene['tonemap']['gamma'] = tch_var_f([1.0])  # Linear output

    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    #z = np.sqrt(1 - np.min(np.stack((x ** 2 + y ** 2, np.ones_like(x)), axis=-1), axis=-1))
    unit_disk_mask = (x ** 2 + y ** 2) <= 1
    z = np.sqrt(1 - unit_disk_mask * (x ** 2 + y ** 2))

    # Make a hemi-sphere bulging out of the xy-plane scene
    z[~unit_disk_mask] = 0
    pos = np.stack((x.ravel(), y.ravel(), z.ravel() - 5, np.ones(num_samples)), axis=1)

    # Normals outside the sphere should be [0, 0, 1]
    x[~unit_disk_mask] = 0
    y[~unit_disk_mask] = 0
    z[~unit_disk_mask] = 1

    normals = np_normalize(np.stack((x.ravel(), y.ravel(), z.ravel(), np.zeros(num_samples)), axis=1))

    if b_display:
        plt.ion()
        plt.figure()
        plt.subplot(131)
        plt.imshow(pos[..., 0].reshape((height, width)))
        plt.subplot(132)
        plt.imshow(pos[..., 1].reshape((height, width)))
        plt.subplot(133)
        plt.imshow(pos[..., 2].reshape((height, width)))

        plt.figure()
        plt.imshow(normals[..., 2].reshape((height, width)))

    ## Convert to the camera's coordinate system
    #Mcam = lookat(eye=large_scene['camera']['eye'], at=large_scene['camera']['at'], up=large_scene['camera']['up'])

    pos_CC = tch_var_f(pos) #torch.matmul(tch_var_f(pos), Mcam.transpose(1, 0))

    large_scene['objects']['disk']['pos'] = pos_CC
    large_scene['objects']['disk']['normal'] = None  # Estimate the normals tch_var_f(normals)
    # large_scene['camera']['eye'] = tch_var_f([-10., 0., 10.])
    # large_scene['camera']['eye'] = tch_var_f([2., 0., 10.])
    large_scene['camera']['eye'] = tch_var_f([-5., 0., 0.])

    # main render run
    start_time = time()
    res = render_splats_along_ray(large_scene, use_quartic=use_quartic)
    rendering_time.append(time() - start_time)

    # Test cam_to_world
    res_world = cam_to_world(res['pos'].reshape(-1, 3), res['normal'].reshape(-1, 3), large_scene['camera'])

    im = get_data(res['image'])
    im = np.uint8(255. * im)

    depth = get_data(res['depth'])
    depth[depth >= large_scene['camera']['far']] = large_scene['camera']['far']

    if b_display:


        plt.figure()
        plt.imshow(im, interpolation='none')
        plt.title('Image')
        plt.savefig(out_dir + '/fig_img_orig.png')

        plt.figure()
        plt.imshow(depth, interpolation='none')
        plt.title('Depth Image')
        #plt.savefig(out_dir + '/fig_depth_orig.png')

        plt.figure()
        pos_world = get_data(res_world['pos'])
        posx_world = pos_world[:, 0].reshape((im.shape[0], im.shape[1]))
        posy_world = pos_world[:, 1].reshape((im.shape[0], im.shape[1]))
        posz_world = pos_world[:, 2].reshape((im.shape[0], im.shape[1]))
        plt.subplot(131)
        plt.imshow(posx_world)
        plt.title('x_world')
        plt.subplot(132)
        plt.imshow(posy_world)
        plt.title('y_world')
        plt.subplot(133)
        plt.imshow(posz_world)
        plt.title('z_world')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pos_world[:, 0], pos_world[:, 1], pos_world[:, 2], s=1.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        plt.figure()
        pos_world = get_data(res['pos'].reshape(-1, 3))
        posx_world = pos_world[:, 0].reshape((im.shape[0], im.shape[1]))
        posy_world = pos_world[:, 1].reshape((im.shape[0], im.shape[1]))
        posz_world = pos_world[:, 2].reshape((im.shape[0], im.shape[1]))
        plt.subplot(131)
        plt.imshow(posx_world)
        plt.title('x_CC')
        plt.subplot(132)
        plt.imshow(posy_world)
        plt.title('y_CC')
        plt.subplot(133)
        plt.imshow(posz_world)
        plt.title('z_CC')

    imsave(out_dir + '/img_orig.png', im)
    #imsave(out_dir + '/depth_orig.png', im_depth)

    # hold matplotlib figure
    plt.ioff()
    plt.show()


def main():
    import os
    import argparse

    parser = argparse.ArgumentParser(usage="render_splat_NDC_demo.py --out_dir output_dir "
                                           "--width 128 --height 128")
    parser.add_argument('--out_dir', type=str, default='./render_samples/', help='Directory for rendered images.')
    parser.add_argument('--width', type=int, default=128, help='Width of output image.')
    parser.add_argument('--height', type=int, default=128, help='Height of output image.')
    parser.add_argument('--fovy', type=float, default=18.0, help='Field of view in the vertical direction.')
    parser.add_argument('--f', type=float, default=0.1, help='Focal length of camera.')
    parser.add_argument('--norm_depth_image_only', action='store_true', default=False, help='Render on the normalized'
                                                                                            ' depth image.')
    parser.add_argument('--test_NDC', action='store_true', help='')
    parser.add_argument('--display', action='store_true', help='Optionally display using matplotlib.')
    parser.add_argument('--use_quartic', action='store_true', help='Use quartic attenuation')

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    #res = render_splats_NDC(SCENE_TEST, norm_depth_image_only=True)
    #print(res)
    #render_sphere_world(out_dir='./test_out', cam_pos=[0, 0, 10], radius=0.03, width=64, height=64,
    #                    fovy=11.5, focal_length=0.01, b_display=True)
    if not args.test_NDC:
        test_sphere_splat_render_along_ray(out_dir=args.out_dir, cam_pos=[0, 0, 10], width=args.width,
                                           height=args.height, fovy=11.5, focal_length=0.01,
                                           use_quartic=args.use_quartic, b_display=True)
    else:
        test_sphere_splat_NDC(out_dir=args.out_dir, cam_pos=[0, 0, 10], width=args.width, height=args.height,
                              fovy=11.5, focal_length=0.01, b_display=True)


if __name__ == '__main__':
    main()
