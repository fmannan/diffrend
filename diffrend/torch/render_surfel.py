import numpy as np
import torch
from diffrend.torch.utils import tch_var_f, tch_var_l, get_data, lookat, lookat_inv, cam_to_world
from diffrend.torch.renderer import render_splats_NDC, render, render_splats_along_ray
from diffrend.numpy.ops import normalize as np_normalize
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
        'ambient': tch_var_f([0.0, 0.0, 0.0]),
        # Light attenuation factors have the form (kc, kl, kq) and eq: 1/(kc + kl * d + kq * d^2)
        'attenuation': tch_var_f([
            [1., 0., 0.0],
            [0., 0., 0.01],
            [0., 0., 0.01],
        ])
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


def test_sphere_splat_render_along_ray(cam_pos, width, height, fovy, focal_length, use_quartic):
    """
    Create a sphere on a square as in render_sphere_world, and then convert to the camera's coordinate system
    and then render using render_splats_along_ray.
    """
    import copy
    print('render sphere along ray')
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
    # z = np.sqrt(1 - np.min(np.stack((x ** 2 + y ** 2, np.ones_like(x)), axis=-1), axis=-1))
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

    plt.ion()

    pos_CC = tch_var_f(pos)  # torch.matmul(tch_var_f(pos), Mcam.transpose(1, 0))

    large_scene['objects']['disk']['pos'] = pos_CC
    large_scene['objects']['disk']['normal'] = tch_var_f(normals)
    # large_scene['camera']['eye'] = tch_var_f([-10., 0., 10.])
    # large_scene['camera']['eye'] = tch_var_f([2., 0., 10.])
    large_scene['camera']['eye'] = tch_var_f([-5., 0., 0.])

    # main render run
    start_time = time()
    res = render_splats_along_ray(large_scene, use_quartic=use_quartic)
    rendering_time.append(time() - start_time)

    im = get_data(res['image'])
    im = np.uint8(255. * im)

    depth = get_data(res['depth'])
    depth[depth >= large_scene['camera']['far']] = large_scene['camera']['far']

    plt.figure()
    plt.imshow(im, interpolation='none')
    plt.title('Image')

    plt.figure()
    plt.imshow(depth, interpolation='none')
    plt.title('Depth Image')

    # hold matplotlib figure
    plt.ioff()
    plt.show()


test_sphere_splat_render_along_ray(cam_pos=[0, 10, 10], width=128,
                                   height=128, fovy=18, focal_length=0.01,
                                   use_quartic=False)
