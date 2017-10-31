import numpy as np
from diffrend.numpy.quaternion import Quaternion


def norm_p(u, p=2):
    return np.sqrt(np.sum(np.abs(u) ** p))


def norm_sqr(u):
    return np.sum(u ** 2)


def norm(u):
    return np.sqrt(norm_sqr(u))


def normalize(u):
    u = np.array(u)
    return u / norm(u)


def rotate_angle_axis(angle, axis, vec):
    pass


def crossprod_matrix(v):
    x, y, z = v
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]])
