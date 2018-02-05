import torch
from diffrend.torch.utils import tch_var_f

def perspective_LH_NO(fovy, aspect, near, far):
    """Left-handed camera with all coords mapped to [-1, 1]
    :param fovy:
    :param aspect:
    :param near:
    :param far:
    :return:
    """
    tanHalfFovy = torch.tan(fovy / 2.)
    mat_00 = 1 / (aspect * tanHalfFovy)
    mat_11 = 1 / tanHalfFovy
    mat_22 = (near + far) / (far - near)
    mat_23 = -2 * near * far / (far - near)

    return tch_var_f([[mat_00, 0, 0, 0],
                     [0, mat_11, 0, 0],
                     [0, 0, mat_22, mat_23],
                     [0, 0, 1, 0]])


def perspective_RH_NO(fovy, aspect, near, far):
    """Right-handed camera with all coords mapped to [-1, 1]
    :param fovy:
    :param aspect:
    :param near:
    :param far:
    :return:
    """
    tanHalfFovy = torch.tan(fovy / 2.)
    mat_00 = 1 / (aspect * tanHalfFovy)
    mat_11 = 1 / tanHalfFovy
    mat_22 = (near + far) / (far - near)
    mat_23 = -2 * near * far / (far - near)

    return tch_var_f([[mat_00, 0, 0, 0],
                     [0, mat_11, 0, 0],
                     [0, 0, -mat_22, mat_23],
                     [0, 0, -1, 0]])


def perspective(fovy, aspect, near, far, type='RH_NO'):
    perspective_fn = {'LH_NO': perspective_LH_NO,
                      'RH_NO': perspective_RH_NO
                      }
    return perspective_fn[type](fovy, aspect, near, far)


def sph2cart(u):
    """
    :param u: N x 3 in [radius, azimuth, inclination]
    :return:
    """
    """
    :param radius:
    :param phi: azimuth, i.e., angle between x-axis and xy proj of vector r * sin(theta)
    :param theta:  inclination, i.e., angle between vector and z-axis
    :return: [x, y, z]
    """
    radius, phi, theta = u[..., 0], u[..., 1], u[..., 2]
    sinth = torch.sin(theta)
    x = sinth * torch.cos(phi) * radius
    y = sinth * torch.sin(phi) * radius
    z = torch.cos(theta) * radius
    return torch.stack((x, y, z), dim=-1)
