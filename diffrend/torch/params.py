import numpy as np
import torch
from diffrend.torch.utils import tch_var_f, tch_var_l

OUTPUT_FOLDER = "output/"


# Starter scene for rendering splats
SCENE_BASIC = {
    'camera': {
        'viewport': [0, 0, 320, 240],
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
            [20., 20., 20., 1.0],
            [-15, 3., 15., 1.0],
            [2, 0., 10., 1.0],
        ]),
        'color_idx': tch_var_l([2, 1, 3]),
        # Light attenuation factors have the form (kc, kl, kq) and eq: 1/(kc + kl * d + kq * d^2)
        'attenuation': [
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
        ]
    },
    'colors': tch_var_f([
        [0.0, 0.0, 0.0],
        [0.8, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.2, 0.8, 0.2],
    ]),
    'materials': {'albedo': tch_var_f([
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.5, 0.5, 0.5],
        [0.9, 0.1, 0.1],
        [0.1, 0.6, 0.8],
    ])
    },
    'objects': {
        'disk': {
            'normal': tch_var_f([
                [0., 0., 1., 0.0],
                [0., 1.0, 0.0, 0.0],
                [-1., -1.0, 1., 0.0]
            ]),
            'pos': tch_var_f([
                [0., -1., 3., 1.0],
                [0., -1., 0, 1.0],
                [10., 5., -5, 1.0]
            ]),
            'radius': tch_var_f([4, 7, 4]),
            'material_idx': tch_var_l([4, 3, 5])
        }
    },
    'tonemap': {
        'type': 'gamma',
        'gamma': tch_var_f([0.8])
    },
}


# Scene with disks and spheres
SCENE_1 = {
    'camera': {
        'viewport': [0, 0, 320, 240],
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
            [20., 20., 20., 1.0],
            [-15, 3., 15., 1.0],
            [2, 0., 10., 1.0],
        ]),
        'color_idx': tch_var_l([2, 1, 3]),
        # Light attenuation factors have the form (kc, kl, kq) and eq: 1/(kc + kl * d + kq * d^2)
        'attenuation': [
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
        ]
    },
    'colors': tch_var_f([
        [0.0, 0.0, 0.0],
        [0.8, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.2, 0.8, 0.2],
    ]),
    'materials': {'albedo': tch_var_f([
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.5, 0.5, 0.5],
        [0.9, 0.1, 0.1],
        [0.1, 0.6, 0.8],
    ])
    },
    'objects': {
        'disk': {
            'normal': tch_var_f([
                [0., 0., 1., 0.0],
                [0., 1.0, 0.0, 0.0],
                [-1., -1.0, 1., 0.0]
            ]),
            'pos': tch_var_f([
                [0., -1., 3., 1.0],
                [0., -1., 0, 1.0],
                [10., 5., -5, 1.0]
            ]),
            'radius': tch_var_f([4, 7, 4]),
            'material_idx': tch_var_l([4, 3, 5])
        },
        'sphere': {'pos': tch_var_f([[-8.0, 4.0, -8.0, 1.0],
                                     [10.0, 0.0, -4.0, 1.0]
                                     ]),
                   'radius': tch_var_f([3.0, 2.0]),
                   'material_idx': tch_var_l([3, 3])
        },
        'triangle': {'face': tch_var_f([[[-20.0, -18.0, -10.0, 1.0],
                                         [10.0, -18.0, -10.0, 1.],
                                         [-2.5, 18.0, -10.0, 1.]],
                                        [[15.0, -18.0, -10.0, 1.0],
                                         [25, -18.0, -10.0, 1.],
                                         [20, 18.0, -10.0, 1.]]
                                        ]),
                     'normal': tch_var_f([[0., 0., 1., 0.],
                                          [0., 0., 1., 0.]
                                          ]),
                     'material_idx': tch_var_l([5, 4])
        },
    },
    'tonemap': {
        'type': 'gamma',
        'gamma': tch_var_f([0.8])
    },
}
