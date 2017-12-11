"""Sample generator."""
from diffrend.model import compute_face_normal
import numpy as np


def uniform_sample_circle(radius, num_samples, normal=np.array([0., 0., 1.])):
    """Generate uniform random samples into a circle."""
    theta = np.random.rand(num_samples) * 2 * np.pi
    return radius * np.stack((np.cos(theta), np.sin(theta),
                              np.zeros_like(theta)), axis=1)


def uniform_sample_cylinder(radius, height, num_samples,
                            normal=np.array([0., 0., 1.])):
    """Generate uniform random samples into a cilinder."""
    theta = np.random.rand(num_samples) * 2 * np.pi
    z = height * (np.random.rand(num_samples) - .5)
    return radius * np.stack((np.cos(theta), np.sin(theta), z), axis=1)


def uniform_sample_sphere(radius, num_samples):
    """Generate uniform random samples into a sphere."""
    pts_2d = np.random.rand(num_samples, 2)
    # theta is angle from the z-axis
    theta = 2 * np.arccos(np.sqrt(1 - pts_2d[:, 0]))
    phi = 2 * np.pi * pts_2d[:, 1]
    pts = np.stack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                    np.cos(theta)), axis=1) * radius
    return pts


# TODO: Unfinished
def uniform_sample_torus(inner_radius, outer_radius, num_samples,
                         normal=np.array([0., 0., 1.])):
    """Rejection samplign based method.

    From: https://math.stackexchange.com/questions/2017079/uniform-random-points-on-a-torus
    Here I use a different one that works in parallel. Need to check if this is
    correct.
    1. First generate samples between inner_radius and outer_radius based on
    probability weighted by [inner_rad, outer_rad]:
        shift to (outer+inner) / 2
    2. uniform randomly choose the sign of z and compute:
        rad = outer - inner
        r * cos(theta) = y => theta = arccos(x/r)
        z = r * sin(theta)
    3. uniformly choose phi, and rotate all the points by R(phi, z)
    """
    r = outer_radius - inner_radius
    R = (inner_radius + outer_radius) / 2.


def uniform_sample_triangle(v, num_samples):
    """Generate uniform random samples into a triangle."""
    samples = np.random.rand(num_samples, 2)
    # surface parameters
    s, t = samples[:, 0], samples[:, 1]
    # barycentric coordinates
    sqrt_s = np.sqrt(s)
    b = np.stack((1 - sqrt_s, (1 - t) * sqrt_s, t * sqrt_s), axis=1)

    if np.ndim(v) == 2:  # single triangle
        assert v.shape[0] == 3 and v.shape[1] == 3
        v = v[np.newaxis, ...]

    # first axis is number of faces
    assert np.ndim(v) == 3 and v.shape[1] == 3 and v.shape[2] == 3

    return np.squeeze(np.sum(b[..., np.newaxis] * v, axis=1))


def triangle_double_area(obj):
    """Triangle double area.

    https://github.com/alecjacobson/gptoolbox/blob/master/mesh/doublearea.m
    :param obj:
    :return:
    """
    v = obj['v']
    f = obj['f']

    v1, v2, v3 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]

    if v.shape[-1] == 2:
        r = v1 - v3
        s = v2 - v3
        dblA = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    elif v.shape[-1] == 3:
        dblA = np.sqrt(triangle_double_area({'v': v[:, [1, 2]], 'f': f}) ** 2 +
                       triangle_double_area({'v': v[:, [2, 0]], 'f': f}) ** 2 +
                       triangle_double_area({'v': v[:, [0, 1]], 'f': f}) ** 2)
    else:
        raise ValueError("Not Implemented")

    return dblA


def uniform_sample_mesh(obj, num_samples):
    """Generate uniform random samples into a mesh."""
    v = obj['v']
    f = obj['f']
    if 'a' in obj:
        area = obj['a']
    else:
        area = triangle_double_area(obj)

    if 'fn' in obj:
        fn = obj['fn']
    else:
        fn = compute_face_normal(obj)

    prob_area = area / np.sum(area)

    # First choose triangles based on their size
    idx = np.random.choice(f.shape[0], num_samples, p=prob_area)
    # Construct batch of triangles
    tri = np.concatenate((v[f[idx, 0]][:, np.newaxis, :],
                          v[f[idx, 1]][:, np.newaxis, :],
                          v[f[idx, 2]][:, np.newaxis]),
                         axis=1)
    vn = fn[idx]
    return uniform_sample_triangle(tri, num_samples), vn


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.ion()

    pts = uniform_sample_sphere(radius=1.0, num_samples=1000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])

    pts = uniform_sample_circle(radius=1.0, num_samples=1000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])

    pts = uniform_sample_cylinder(radius=0.25, height=1.0, num_samples=1000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])

    v = np.array([[0., 0., 0.], [1., 0., 0.], [0.5, 1.0, 0.]])
    pts = uniform_sample_triangle(v, num_samples=1000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    plt.xlabel('x')
    plt.ylabel('y')

    plt.figure()
    plt.plot(pts[:, 0], pts[:, 1], 'r.')

    v = np.array([[0., 0., 0.], [1., 0., 0.], [0.5, 1.0, 0.], [2., 2., 1.]])
    f = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32)
    pts, vn = uniform_sample_mesh({'v': v, 'f': f}, num_samples=1000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    plt.xlabel('x')
    plt.ylabel('y')

    from diffrend.model import load_model

    obj = load_model('../../data/chair_0001.off')
    pts, vn = uniform_sample_mesh(obj, num_samples=1000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.6)
    plt.xlabel('x')
    plt.ylabel('y')

    obj = load_model('../../data/bunny.obj')
    pts_obj, vn = uniform_sample_mesh(obj, num_samples=800)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts_obj[:, 0], pts_obj[:, 1], pts_obj[:, 2])
    ax.view_init(93, -64)
    plt.xlabel('x')
    plt.ylabel('y')

    obj = load_model('../../data/desk_0007.off')
    pts_obj, vn = uniform_sample_mesh(obj, num_samples=1000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts_obj[:, 0], pts_obj[:, 1], pts_obj[:, 2])
    plt.xlabel('x')
    plt.ylabel('y')

    plt.ioff()
    plt.show()
