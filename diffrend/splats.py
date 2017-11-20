"""Splats module."""
import numpy as np
import random
from diffrend.numpy_.renderer import render_scene


class Scene(object):
    """Class to save a scene."""

    def __init__(self, n_lights, n_splats):
        """Scene constructor."""
        self.n_lights = n_lights
        self.n_splats = n_splats

        # Constants
        self.n_param_camera = 20
        self.n_param_tonemap = 2
        self.n_param_light = 9
        self.n_param_splat = 10

        self.scene = {'camera': {'viewport': [],
                                 'fovy': 0.0,
                                 'focal_length': 0.0,
                                 'eye': [],
                                 'up': [],
                                 'at': [],
                                 'near': 0.0,
                                 'far': 0.0},
                      'lights': {'pos': np.zeros(shape=(n_lights, 4)),
                                 'color_idx': np.zeros(shape=(n_lights,),
                                                       dtype=int),
                                 'attenuation': np.zeros(shape=(n_lights, 3))},
                      'colors': np.zeros(shape=(n_lights, 3)),
                      'materials': {'albedo': np.zeros(shape=(n_splats, 3))},
                      'objects': {'disk':
                                  {'normal': np.zeros(shape=(n_splats, 4)),
                                   'pos': np.zeros(shape=(n_splats, 4)),
                                   'radius': np.zeros(shape=(n_splats,)),
                                   'material_idx': np.zeros(shape=(n_splats,),
                                                            dtype=int)}
                                  },
                      'tonemap': {'type': '', 'gamma': 0.0}
                      }

    def set_camera(self, viewport, eye, up, at, fovy=90.0, focal_length=1.0,
                   near=1.0, far=1000.0):
        """Set camera parameters."""
        assert viewport.shape == (4,)
        assert eye.shape == (4,)
        assert up.shape == (4,)
        assert at.shape == (4,)
        assert isinstance(fovy, float)
        assert isinstance(focal_length, float)
        assert isinstance(near, float)
        assert isinstance(far, float)

        self.scene['camera']['viewport'] = viewport
        self.scene['camera']['eye'] = eye
        self.scene['camera']['up'] = up
        self.scene['camera']['at'] = at
        self.scene['camera']['fovy'] = np.deg2rad(fovy)
        self.scene['camera']['focal_length'] = focal_length
        self.scene['camera']['near'] = near
        self.scene['camera']['far'] = far

    def set_camera_array(self, data):
        """Set camera parameters from an array."""
        assert data.shape == (self.n_param_camera,)
        self.set_camera(
         data[0:4].astype(int), data[4:8], data[8:12], data[12:16],
         data[16], data[17], data[18], data[19])

    def set_tonemap(self, tonemap_type, gamma):
        """Set tonemap."""
        assert isinstance(tonemap_type, str)
        assert isinstance(gamma, float)
        self.scene['tonemap']['type'] = tonemap_type
        self.scene['tonemap']['gamma'] = gamma

    def set_light(self, id, pos, color, attenuation):
        """Add a light to the list."""
        assert pos.shape == (3,)
        assert color.shape == (3,)
        assert attenuation.shape == (3,)

        self.scene['colors'][id] = color
        self.scene['lights']['pos'][id] = np.append(pos, 1.0)
        self.scene['lights']['color_idx'][id] = id
        self.scene['lights']['attenuation'][id] = attenuation

    def set_light_array(self, id, data):
        """Set light parameters from an array."""
        assert data.shape == (self.n_param_light,)
        self.set_light(id, data[0:3], data[3:6], data[6:9])

    def set_lights(self, data):
        """Set all the light parameters from an array."""
        assert data.shape == (self.n_param_light*self.n_lights,)
        for i in range(self.n_lights):
            self.set_light_array(
                i, data[i*self.n_param_light:self.n_param_light*(i+1)])

    def set_splat(self, id, normal, pos, radius, albedo):
        """Add a splat to the list."""
        assert normal.shape == (3,)
        assert pos.shape == (3,)
        assert isinstance(radius, float)
        assert albedo.shape == (3,)

        self.scene['materials']['albedo'][id] = albedo
        self.scene['objects']['disk']['normal'][id] = np.append(normal, 1.0)
        self.scene['objects']['disk']['pos'][id] = np.append(pos, 0.0)
        self.scene['objects']['disk']['radius'][id] = radius
        self.scene['objects']['disk']['material_idx'][id] = id

    def set_splat_array(self, id, data):
        """Set splat parameters from an array."""
        assert data.shape == (self.n_param_splat,)
        self.set_splat(id, data[0:3], data[3:6], data[6], data[7:10])

    def set_splats(self, data):
        """Set all the splat parameters from an array."""
        assert data.shape == (self.n_param_splat*self.n_splats,)
        for i in range(self.n_splats):
            self.set_splat_array(
                i, data[i*self.n_param_splat:self.n_param_splat*(i+1)])

    def __str__(self):
        """Magic method for convert a scene into a string. Just for print."""
        return ''.join(("CAMERA: ", str(self.scene['camera']),
                        "\nTONEMAP: ", str(self.scene['tonemap']),
                        "\nLIGHTS: ", str(self.scene['lights']),
                        "\nCOLORS: ", str(self.scene['colors']),
                        "\nMATERIALS: ", str(self.scene['materials']),
                        "\nSPLATS: ", str(self.scene['objects'])))


def create_sample_scene():
    """Create a sample scene."""
    scene = Scene(n_lights=2, n_splats=3)

    # Camera
    scene.set_camera(viewport=np.array([0, 0, 320, 240]),
                     eye=np.array([0.0, 1.0, 10.0, 1.0]),
                     up=np.array([0.0, 1.0, 0.0, 0.0]),
                     at=np.array([0.0, 0.0, 0.0, 1.0]),
                     fovy=90.0,
                     focal_length=1.0,
                     near=1.0,
                     far=1000.0)

    # Tonemap
    scene.set_tonemap(tonemap_type='gamma',
                      gamma=0.8)

    # Lights
    scene.set_light(id=0,
                    pos=np.array([20., 20., 20.]),
                    color=np.array([0.8, 0.1, 0.1]),
                    attenuation=np.array([0.2, 0.2, 0.2]))

    scene.set_light(id=1,
                    pos=np.array([-15, 3., 15.]),
                    color=np.array([0.8, 0.1, 0.1]),
                    attenuation=np.array([0., 1., 0.]))

    # Splats
    scene.set_splat(id=0,
                    normal=np.array([0., 0., 1.]),
                    pos=np.array([0., -1., 3.]),
                    radius=4.0,
                    albedo=np.array([0.9, 0.1, 0.1]))

    scene.set_splat(id=1,
                    normal=np.array([0., 1.0, 0.0]),
                    pos=np.array([0., -1., 0]),
                    radius=7.0,
                    albedo=np.array([0.5, 0.5, 0.5]))

    scene.set_splat(id=2,
                    normal=np.array([-1., -1.0, 1.]),
                    pos=np.array([10., 5., -5]),
                    radius=4.0,
                    albedo=np.array([0.1, 0.1, 0.8]))

    return scene


def create_sample_scene2():
    """Create a sample scene."""
    scene = Scene(n_lights=2, n_splats=3)

    # Camera
    scene.set_camera_array(np.array([0, 0, 320, 240, 0.0, 1.0, 10.0, 1.0,
                                     0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     1.0, 90.0, 1.0, 1.0, 1000.0]))

    # Tonemap
    scene.set_tonemap(tonemap_type='gamma',
                      gamma=0.8)

    # Lights
    scene.set_light_array(0, np.array([20., 20., 20., 0.8, 0.1, 0.1, 0.2,
                                       0.2, 0.2]))
    scene.set_light_array(1, np.array([-15, 3., 15., 0.8, 0.1, 0.1, 0.,
                                       1., 0.]))

    # Splats
    scene.set_splat_array(0, np.array([0., 0., 1., 0., -1., 3., 4.0, 0.9, 0.1,
                                       0.1]))
    scene.set_splat_array(1, np.array([0., 1.0, 0.0, 0., -1., 0, 7.0, 0.5, 0.5,
                                       0.5]))
    scene.set_splat_array(2, np.array([-1., -1.0, 1., 10., 5., -5, 4.0, 0.1,
                                       0.1, 0.8]))
    return scene


def create_sample_scene3():
    """Create a sample scene."""
    scene = Scene(n_lights=2, n_splats=3)

    # Camera
    scene.set_camera_array(np.array([0, 0, 320, 240, 0.0, 1.0, 10.0, 1.0,
                                     0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     1.0, 90.0, 1.0, 1.0, 1000.0]))

    # Tonemap
    scene.set_tonemap(tonemap_type='gamma', gamma=0.8)

    # Lights
    scene.set_lights(np.array([20., 20., 20., 0.8, 0.1, 0.1, 0.2, 0.2,
                               0.2, -15, 3., 15., 0.8, 0.1, 0.1, 0., 1.,
                               0.]))

    # Splats
    scene.set_splats(np.array([0., 0., 1., 0., -1., 3., 4.0, 0.9, 0.1, 0.1,
                               0., 1.0, 0.0, 0., -1., 0, 7.0, 0.5, 0.5, 0.5,
                               -1., -1.0, 1., 10., 5., -5, 4.0, 0.1, 0.1,
                               0.8]))
    return scene


def create_random_scene(n_lights=2, n_splats=3, max_radius=5,
                        max_splat_pos=5.0, max_light_pos=20.0):
    """Create a sample scene."""
    scene = Scene(n_lights, n_splats)

    # Camera
    scene.set_camera(viewport=np.array([0, 0, 320, 240]),
                     eye=np.array([0.0, 1.0, 10.0, 1.0]),
                     up=np.array([0.0, 1.0, 0.0, 0.0]),
                     at=np.array([0.0, 0.0, 0.0, 1.0]),
                     fovy=90.0,
                     focal_length=1.0,
                     near=1.0,
                     far=1000.0)

    # Tonemap
    scene.set_tonemap(tonemap_type='gamma', gamma=0.8)

    # Lights
    for i in range(n_lights):
        scene.set_light(i,
                        pos=np.random.rand(3)*max_light_pos*2-max_light_pos,
                        color=np.random.rand(3),
                        attenuation=np.random.rand(3))

    # Splats
    for i in range(n_splats):
        scene.set_splat(i,
                        normal=np.random.rand(3)*2-1,
                        pos=np.random.rand(3)*max_splat_pos*2-max_splat_pos,
                        radius=random.random()*max_radius,
                        albedo=np.random.rand(3))
    return scene


def main():
    """Test function."""
    # Create Scene
    # scene = create_random_scene(10, 3)
    scene = create_random_scene()
    print (scene)

    # Render Scene
    render_scene(scene.scene)


if __name__ == '__main__':
    main()
