import numpy as np
import torch
from diffrend.torch.utils import tch_var_f, tch_var_l_ng

OUTPUT_FOLDER = "output/"


# everything in here has to be differentiated for
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
            [-15, 3., 15., 1.0]
        ]),
        'color_idx': tch_var_l_ng([2, 1]),
        # Light attenuation factors have the form (kc, kl, kq) and eq: 1/(kc + kl * d + kq * d^2)
        'attenuation': [
            [0., 1., 0.],
            [0., 0., 1.]
        ]
    },
    'colors': tch_var_f([
        [0.0, 0.0, 0.0],
        [0.8, 0.1, 0.1],
        [0.2, 0.2, 0.2]
    ]),
    'materials': {'albedo': tch_var_f([
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.5, 0.5, 0.5],
        [0.9, 0.1, 0.1],
        [0.1, 0.1, 0.8],
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
            'material_idx': tch_var_l_ng([4, 3, 5])
        }
    },
    'tonemap': {
        'type': 'gamma',
        'gamma': 0.8
    },
}
