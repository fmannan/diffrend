import numpy as np
from diffrend.numpy.vector import Vector
from diffrend.numpy.quaternion import Quaternion
import diffrend.numpy.ops as ops

def lookat(eye, at, up):
    """Returns a lookat matrix

    :param eye:
    :param at:
    :param up:
    :return:
    """
    if type(eye) is list:
        eye = np.array(eye, dtype=np.float32)
    if type(at) is list:
        at = np.array(at, dtype=np.float32)
    if type(up) is list:
        up = np.array(up, dtype=np.float32)

    if up.size == 4:
        assert up[3] == 0
        up = up[:3]

    z = (eye - at)
    z = (z / np.linalg.norm(z, 2))[:3]

    y = up / np.linalg.norm(up, 2)
    x = np.cross(y, z)

    matrix = np.eye(4)
    matrix[:3, :3] = np.stack((x, y, z), axis=1).T

    return matrix


class Camera(object):
    def __init__(self, pos, orientation, viewport):
        assert isinstance(orientation, Quaternion)
        self.pos = np.array(pos, dtype=np.float32)

        if self.pos.size == 3:
            self.pos = np.append(self.pos, 1.0)

        self.orientation = orientation
        self.viewport = viewport
        self.view_matrix = np.eye(4)

    def __str__(self):
        return 'Camera: pos {}, orientation: {}'.format(self.pos, self.orientation)

    @property
    def M(self):
        return self.view_matrix

    def rotate(self, axis, angle):
        self.orientation = self.orientation.rotate(angle_rad=angle, axis=axis)

    def translate(self, translation):
        self.pos[:3] += translation[:3]

    def lookat(self, eye, at, up):
        """Same as the global lookat but changes the state of the current camera
        :param eye:
        :param at:
        :param up:
        :return:
        """
        if type(eye) is list:
            eye = np.array(eye, dtype=np.float32)
        if eye.size == 3:
            eye = np.append(eye, 1.0)
        self.pos = eye
        self.view_matrix = lookat(self.pos, at, up)

    def generate_rays(self):
        pass


class PinholeCamera(Camera):
    def __init__(self, pos, orientation, fov, focal_length, viewport):
        super(PinholeCamera, self).__init__(pos, orientation, viewport)
        self.fov = float(fov)
        self.focal_length = float(focal_length)
        self.proj_matrix = np.array([[self.focal_length, 0, 0, 0],
                                    [0, self.focal_length, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

    @property
    def M(self):
        return np.dot(self.proj_matrix, self.model_view)

    @property
    def perspective(self):
        return self.proj_matrix

    @property
    def model_view(self):
        w = 1
        if self.pos.size == 4:
            w = self.pos[3]
        translation_matrix = np.array([[1, 0, 0, -self.pos[0] / w],
                                       [0, 1, 0, -self.pos[1] / w],
                                       [0, 0, 1, -self.pos[2] / w],
                                       [0, 0, 0, 1]])
        return np.dot(translation_matrix, self.orientation.R, )


class TrackBallCamera(PinholeCamera):
    def __init__(self, pos, orientation, fov, focal_length, viewport):
        super(TrackBallCamera, self).__init__(pos, orientation, fov, focal_length, viewport)

    def mouse_press(self, coords):
        self.src = ops.normalize([coords[0], coords[1], self.pos[2] - self.focal_length])

    def mouse_move(self, coords):
        self.dst = ops.normalize([coords[0], coords[1], self.pos[2] - self.focal_length])
        print('src', self.src, 'dst:', self.dst)
        # compute object rotation
        axis = np.cross(self.src, self.dst)
        theta = np.arccos(np.dot(self.src, self.dst))
        self.rotate(axis=axis, angle=theta)

        self.src = self.dst
        self.view_matrix = self.orientation.R

    def zoom(self, amount):
        self.translate(np.array([0, 0, amount]))


if __name__ == '__main__':
    cam = TrackBallCamera([0.0, 0.0, 0.0, 1.0], Quaternion(coeffs=[0, 0, 0, 1]), fov=45, focal_length=2,
                          viewport=[0, 0, 640, 480])
