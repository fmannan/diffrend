import numpy as np
import torch

OUTPUT_FOLDER = "output/"

class Params(object):
    def __init__(self, utils):
        self.utils = utils

        # Starter scene for rendering splats
        self.SCENE_BASIC = {
            'camera': {
                'viewport': [0, 0, 320, 240],
                'fovy': np.deg2rad(90.),
                'focal_length': 1.,
                'eye': self.utils.tch_var_f([0.0, 1.0, 10.0, 1.0]),
                'up': self.utils.tch_var_f([0.0, 1.0, 0.0, 0.0]),
                'at': self.utils.tch_var_f([0.0, 0.0, 0.0, 1.0]),
                'near': 1.0,
                'far': 1000.0,
            },
            'lights': {
                'pos': self.utils.tch_var_f([
                    [10., 0., 0., 1.0],
                    [-10, 0., 0., 1.0],
                    [0, 10., 0., 1.0],
                    [0, -10., 0., 1.0],
                    [0, 0., 10., 1.0],
                    [0, 0., -10., 1.0],
                ]),
                'color_idx': self.utils.tch_var_l([1, 3, 4, 5, 6, 7]),
                # Light attenuation factors have the form (kc, kl, kq) and eq: 1/(kc + kl * d + kq * d^2)
                'attenuation': [
                    [0., 1., 0.],
                    [0., 0., 1.],
                    [1., 0., 0.],
                    [1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.],
                ]
            },
            'colors': self.utils.tch_var_f([
                [0.0, 0.0, 0.0],
                [0.8, 0.1, 0.1],
                [0.2, 0.2, 0.2],
                [0.2, 0.8, 0.2],
                [0.2, 0.2, 0.8],
                [0.8, 0.2, 0.8],
                [0.8, 0.8, 0.2],
                [0.2, 0.8, 0.8],
            ]),
            'materials': {'albedo': self.utils.tch_var_f([
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
                    'normal': self.utils.tch_var_f([
                        [0., 0., 1., 0.0],
                        [0., 1.0, 0.0, 0.0],
                        [-1., -1.0, 1., 0.0]
                    ]),
                    'pos': self.utils.tch_var_f([
                        [0., -1., 3., 1.0],
                        [0., -1., 0, 1.0],
                        [10., 5., -5, 1.0]
                    ]),
                    'radius': self.utils.tch_var_f([4, 7, 4]),
                    'material_idx': self.utils.tch_var_l([4, 3, 5])
                }
            },
            'tonemap': {
                'type': 'gamma',
                'gamma': self.utils.tch_var_f([0.8])
            },
        }
        
        
        # Scene with disks and spheres
        self.SCENE_1 = {
            'camera': {
                'viewport': [0, 0, 320, 240],
                'fovy': np.deg2rad(90.),
                'focal_length': 1.,
                'eye': self.utils.tch_var_f([0.0, 1.0, 10.0, 1.0]),
                'up': self.utils.tch_var_f([0.0, 1.0, 0.0, 0.0]),
                'at': self.utils.tch_var_f([0.0, 0.0, 0.0, 1.0]),
                'near': 1.0,
                'far': 1000.0,
            },
            'lights': {
                'pos': self.utils.tch_var_f([
                    [20., 20., 20., 1.0],
                    [-15, 3., 15., 1.0],
                    [2, 0., 10., 1.0],
                ]),
                'color_idx': self.utils.tch_var_l([2, 1, 3]),
                # Light attenuation factors have the form (kc, kl, kq) and eq: 1/(kc + kl * d + kq * d^2)
                'attenuation': [
                    [0., 1., 0.],
                    [0., 0., 1.],
                    [1., 0., 0.],
                ]
            },
            'colors': self.utils.tch_var_f([
                [0.0, 0.0, 0.0],
                [0.8, 0.1, 0.1],
                [0.2, 0.2, 0.2],
                [0.2, 0.8, 0.2],
            ]),
            'materials': {'albedo': self.utils.tch_var_f([
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
                    'normal': self.utils.tch_var_f([
                        [0., 0., 1., 0.0],
                        [0., 1.0, 0.0, 0.0],
                        [-1., -1.0, 1., 0.0],
                        [0., 0., -1., 0.0],
                    ]),
                    'pos': self.utils.tch_var_f([
                        [0., -1., 3., 1.0],
                        [0., -1., 0, 1.0],
                        [10., 5., -5, 1.0],
                        [-10, -8., -5, 1.0],
                    ]),
                    'radius': self.utils.tch_var_f([4, 7, 4, 3]),
                    'material_idx': self.utils.tch_var_l([4, 3, 5, 3])
                },
                'sphere': {'pos': self.utils.tch_var_f([[-8.0, 4.0, -8.0, 1.0],
                                             [10.0, 0.0, -4.0, 1.0]
                                             ]),
                           'radius': self.utils.tch_var_f([3.0, 2.0]),
                           'material_idx': self.utils.tch_var_l([3, 3])
                },
                'triangle': {'face': self.utils.tch_var_f([[[-20.0, -18.0, -10.0, 1.0],
                                                 [10.0, -18.0, -10.0, 1.],
                                                 [-2.5, 18.0, -10.0, 1.]],
                                                [[15.0, -18.0, -10.0, 1.0],
                                                 [25, -18.0, -10.0, 1.],
                                                 [20, 18.0, -10.0, 1.]]
                                                ]),
                             'normal': self.utils.tch_var_f([[0., 0., 1., 0.],
                                                  [0., 0., 1., 0.]
                                                  ]),
                             'material_idx': self.utils.tch_var_l([5, 4])
                },
            },
            'tonemap': {
                'type': 'gamma',
                'gamma': self.utils.tch_var_f([0.8])
            },
        }
