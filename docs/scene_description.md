# Scene Description File Format

All vectors are in homogeneous coordinates (i.e., 4D with last entry
non-zero for finite points and 0 for points at infinity).

Camera, lighting and geometry are specified in the scene description file
as follows.
### Example
Also see `data/basic_scene.json`.
```python
scene_basic = {'camera': {'viewport': [0, 0, 320, 240],
                           'fovy': np.deg2rad(90.),
                           'focal_length': 1.,
                           'eye': [0.0, 0.0, 10.0, 1.0],
                           'up': [0.0, 1.0, 0.0, 0.0],
                           'at': [0.0, 0.0, 0.0, 1.0],
                           'near': 1.0,
                           'far': 1000.0,
                         },
               'lights': {
                   'pos': np.array([[20., 20., 20., 1.0],
                                    [-15, 3., 15., 1.0],
                                    ]),
                   'color_idx': np.array([3, 1]),
                   # Light attenuation factors have the form (kc, kl, kq)
                   # and eq: 1/(kc + kl * d + kq * d^2)
                   'attenuation': np.array([[0., 1., 0.],
                                            [0., 0., 1.]])
               },
               'colors': np.array([[0.0, 0.0, 0.0],
                                   [0.8, 0.1, 0.1],
                                   [0.2, 0.2, 0.2],
                                   [0.1, 0.8, 0.1]
                                   ]),
               'materials': {'albedo': np.array([[0.0, 0.0, 0.0],
                                                 [0.1, 0.1, 0.1],
                                                 [0.2, 0.2, 0.2],
                                                 [0.5, 0.5, 0.5],
                                                 [0.9, 0.1, 0.1],
                                                 [0.1, 0.1, 0.8],
                                                 ]),
                             'trainable': True
                             },
               'objects': {
                   'disk': {
                       'normal': np.array([[0., 0., 1., 0.0],
                                           [0., 1.0, 0.0, 0.0],
                                           [-1., -1.0, 1., 0.0]]),
                       'pos': np.array([[0., -1., 3., 1.0],
                                        [0., -1., 0, 1.0],
                                        [10., 5., -5, 1.0]]),
                       'radius': np.array([4, 7, 4]),
                       'material_idx': np.array([4, 3, 5])
                   },
                   'sphere': {'pos': np.array([[-8.0, 4.0, -8.0, 1.0],
                                               [10.0, 0.0, -4.0, 1.0]]),
                              'radius': np.array([3.0, 2.0]),
                              'material_idx': np.array([3, 3])
                              },
                   'triangle': {'faces': np.array([[[-20.0, -18.0, -10.0, 1.0],
                                                    [10.0, -18.0, -10.0, 1.],
                                                    [-2.5, 18.0, -10.0, 1.]],
                                                   [[15.0, -18.0, -10.0, 1.0],
                                                    [25, -18.0, -10.0, 1.],
                                                    [20, 18.0, -10.0, 1.]]
                                                   ]),
                                'normal': np.array([[0., 0., 1., 0.],
                                                    [0., 0., 1., 0.]]),
                                'material_idx': np.array([5, 4])
                                }
               },
               'tonemap': {'type': 'gamma', 'gamma': 0.8},
               }
```
## Camera
Requires camera position, look-at point, up vector
## Lights

## Materials

## Geometry

### Plane
```python
'plane': {
           'normal': np.array([[0., 0., 1., 0.0],
                               [0., 1.0, 0.0, 0.0],
                               [-1., -1.0, 1., 0.0]]),
           'pos': np.array([[0., -1., 3., 1.0],
                            [0., -1., 0, 1.0],
                            [10., 5., -5, 1.0]]),
           'material_idx': np.array([4, 3, 5])
       },
```

### Disk
Requires position, normal and radius.
```python
'disk': {
           'normal': np.array([[0., 0., 1., 0.0],
                               [0., 1.0, 0.0, 0.0],
                               [-1., -1.0, 1., 0.0]]),
           'pos': np.array([[0., -1., 3., 1.0],
                            [0., -1., 0, 1.0],
                            [10., 5., -5, 1.0]]),
           'radius': np.array([4, 7, 4]),
           'material_idx': np.array([4, 3, 5])
       },
```

