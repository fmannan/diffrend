import numpy as np

OUTPUT_FOLDER = "output/"


# everything in here has to be differentiated for
SCENE_BASIC = {
    'camera': {
        'viewport': [0, 0, 320, 240],
        'fovy': np.deg2rad(90.),
        'focal_length': 1.,
        'eye': [0.0, 0.0, 10.0, 1.0],
        'up': [0.0, 1.0, 0.0, 0.0],
        'at': [0.0, 0.0, 0.0, 1.0],
        'near': 1.0,
        'far': 1000.0,
    },
    'lights': {
        'pos': [
            [20., 20., 20., 1.0],
            [-15, 3., 15., 1.0]
        ],
        'color_idx': [2, 1],
        # Light attenuation factors have the form (kc, kl, kq) and eq: 1/(kc + kl * d + kq * d^2)
        'attenuation': [
            [0., 1., 0.],
            [0., 0., 1.]
        ]
    },
    'colors': [
        [0.0, 0.0, 0.0],
        [0.8, 0.1, 0.1],
        [0.2, 0.2, 0.2]
    ],
    'materials': {'albedo': [
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.5, 0.5, 0.5],
        [0.9, 0.1, 0.1],
        [0.1, 0.1, 0.8],
    ]
    },
    'objects': {
        'disk': {
            'normal': [
                [0., 0., 1., 0.0],
                [0., 1.0, 0.0, 0.0],
                [-1., -1.0, 1., 0.0]
            ],
            'pos': [
                [0., -1., 3., 1.0],
                [0., -1., 0, 1.0],
                [10., 5., -5, 1.0]
            ],
            'radius': [4, 7, 4],
            'material_idx': [4, 3, 5]
        }
    },
    'tonemap': {
        'type': 'gamma',
        'gamma': 0.8
    },
}
