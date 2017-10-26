import numpy as np
from diffrend.numpy.quaternion import Quaternion


def rotate_angle_axis(angle, axis, vec):
    pass


def crossprod_matrix(v):
    x, y, z = v
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]])
