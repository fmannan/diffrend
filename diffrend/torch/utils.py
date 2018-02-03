import numpy as np
import torch
from torch.autograd import Variable
# TODO: SPLIT THIS INTO UTILS AND OPS (like diffrend.numpy)
CPU_ONLY = False
if torch.cuda.is_available() and not CPU_ONLY:
    CUDA = True
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    CUDA = False
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor

print('CUDA support ', CUDA)

tch_var = lambda x, fn_type, req_grad: Variable(fn_type(x),
                                                requires_grad=req_grad)
tch_var_f = lambda x: tch_var(x, FloatTensor, False)
tch_var_l = lambda x: tch_var(x, LongTensor, False)


def np_var(x, req_grad=False):
    """Convert a numpy into a pytorch variable."""
    if CUDA:
        return Variable(torch.from_numpy(x), requires_grad=req_grad).cuda()
    else:
        return Variable(torch.from_numpy(x), requires_grad=req_grad)

# np_var_f = lambda x: np_var(x, FloatTensor, False)
# np_var_l = lambda x: np_var(x, LongTensor, False)


def get_data(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def bincount(x, nbins, ret_var=True):
    data = x.cpu().data if x.is_cuda else x.data
    count = torch.zeros(nbins).scatter_add_(0, data, torch.ones(x.size()))
    if ret_var:
        count = torch.autograd.Variable(count)
    return count


def where(cond, x, y):
    return cond.float() * x + (1 - cond.float()) * y


def norm_p(u, p=2):
    return torch.pow(torch.sum(torch.pow(u, p), dim=-1), 1./p)


def tensor_cross_prod(u, M):
    """
    :param u:  N x 3
    :param M: N x P x 3
    :return:
    """
    s0 = u[:, 1][:, np.newaxis] * M[..., 2] - u[:, 2][:, np.newaxis] * M[..., 1]
    s1 = -u[:, 0][:, np.newaxis] * M[..., 2] + u[:, 2][:, np.newaxis] * M[..., 0]
    s2 = u[:, 0][:, np.newaxis] * M[..., 1] - u[:, 1][:, np.newaxis] * M[..., 0]

    return torch.stack((s0, s1, s2), dim=2)


def nonzero_divide(x, y):
    """ x and y need to have the same dimensions.
    :param x:
    :param y:
    :return:
    """
    assert list(x.size()) == list(y.size())

    mask = torch.abs(y) > 0
    return x.masked_scatter(mask, x.masked_select(mask) / y.masked_select(mask))


def normalize(u):
    denom = norm_p(u, 2)
    if u.dim() > 1:
        denom = denom[:, np.newaxis]
    # TODO: nonzero_divide for rows with norm = 0
    return u / denom


def point_along_ray(eye, ray_dir, ray_dist):
    """Find the point along the ray_dir at distance ray_dist
    :param eye: 4-element vector or 3-element vector
    :param ray_dir: [4 x N] matrix with N rays and each ray being [x, y, z, 0] direction or [3 x N]
    :param ray_dist: [M x N] matrix with M objects and N rays
    :return: [M x N x 4] intersection points or [M x N x 3]
    """
    return eye[np.newaxis, np.newaxis, :] + ray_dist[:, :, np.newaxis] * ray_dir.transpose(1, 0)[np.newaxis, ...]


def ray_sphere_intersection(eye, ray_dir, sphere, **kwargs):
    """Bundle of rays intersected with a batch of spheres
    :param eye:
    :param ray_dir:
    :param sphere:
    :return:
    """
    pos = sphere['pos'][:, :3]
    pos_tilde = eye[:3] - pos
    radius = sphere['radius']

    a = torch.sum(ray_dir ** 2, dim=0)
    b = 2 * torch.matmul(pos_tilde, ray_dir)
    c = (torch.sum(pos_tilde ** 2, dim=1) - radius ** 2)[:, np.newaxis]

    d_sqr = b ** 2 - 4 * a * c
    intersection_mask = d_sqr >= 0

    d_sqr = where(intersection_mask, d_sqr, 0)

    d = torch.sqrt(d_sqr)
    inv_denom = 1. / (2 * a)

    t1 = (-b - d) * inv_denom
    t2 = (-b + d) * inv_denom

    # get the nearest positive depth
    max_val = torch.max(torch.max(t1, t2)) + 1
    t1 = where(intersection_mask * (t1 >= 0), t1, max_val)
    t2 = where(intersection_mask * (t2 >= 0), t2, max_val)

    ray_dist, _ = torch.min(torch.stack((t1, t2), dim=2), dim=2)
    ray_dist = where(intersection_mask, ray_dist, 1001)

    intersection_pts = point_along_ray(eye, ray_dir, ray_dist)

    normals = intersection_pts - pos[:, np.newaxis, :]
    normals /= torch.sqrt(torch.sum(normals ** 2, dim=-1))[:, :, np.newaxis]

    return {'intersect': intersection_pts, 'normal': normals, 'ray_distance': ray_dist,
            'intersection_mask': intersection_mask}


def ray_plane_intersection(eye, ray_dir, plane, **kwargs):
    """Intersection a bundle of rays with a batch of planes
    :param eye: Camera's center of projection
    :param ray_dir: Ray direction
    :param plane: Plane specification
    :return:
    """
    pos = plane['pos'][:, :3]
    normal = normalize(plane['normal'][:, :3])
    dist = torch.sum(pos * normal, dim=1)

    denom = torch.mm(normal, ray_dir)

    # check for denom = 0
    intersection_mask = torch.abs(denom) > 0

    ray_dist = (dist.unsqueeze(-1) - torch.mm(normal, eye.unsqueeze(-1))) / denom

    intersection_pts = point_along_ray(eye, ray_dir, ray_dist)

    if 'disable_normals' in kwargs and kwargs['disable_normals']:
        normals = None
    else:
        # TODO: Delay this normal selection to save memory. Because even if there's an intersection it may not be visible
        normals = normal[:, np.newaxis, :].repeat(1, intersection_pts.size()[1], 1)

    return {'intersect': intersection_pts, 'normal': normals, 'ray_distance': ray_dist,
            'intersection_mask': intersection_mask}


def ray_disk_intersection(eye, ray_dir, disks, **kwargs):
    result = ray_plane_intersection(eye, ray_dir, disks, **kwargs)
    intersection_pts = result['intersect']
    normals = result['normal']
    ray_dist = result['ray_distance']

    centers = disks['pos'][:, :3]
    radius = disks['radius']
    dist_sqr = torch.sum((intersection_pts - centers[:, np.newaxis, :]) ** 2, dim=-1)

    # Intersection mask
    intersection_mask = (dist_sqr <= radius[:, np.newaxis] ** 2)
    ray_dist = where(intersection_mask, ray_dist, 1001)

    return {'intersect': intersection_pts, 'normal': normals, 'ray_distance': ray_dist,
            'intersection_mask': intersection_mask}


def ray_triangle_intersection(eye, ray_dir, triangles, **kwargs):
    """Intersection of a bundle of rays with a batch of triangles.
    Assumes that the triangles vertices are specified as F x 3 x 4 matrix where F is the number of faces and
    the normals for all faces are precomputed and in a matrix of size F x 4 (i.e., similar to the normals for other
    geometric primitives). Note that here the number of faces F is the same as number of primitives M.
    :param eye:
    :param ray_dir:
    :param triangles:
    :return:
    """

    planes = {'pos': triangles['face'][:, 0, :], 'normal': triangles['normal']}
    result = ray_plane_intersection(eye, ray_dir, planes)
    intersection_pts = result['intersect']  # M x N x 4 matrix where M is the number of objects and N pixels.
    normals = result['normal'][..., :3]  # M x N x 4
    ray_dist = result['ray_distance']

    # check if intersection point is inside or outside the triangle
    # M x N x 3
    v_p0 = (intersection_pts - triangles['face'][:, 0, :3][:, np.newaxis, :])
    v_p1 = (intersection_pts - triangles['face'][:, 1, :3][:, np.newaxis, :])
    v_p2 = (intersection_pts - triangles['face'][:, 2, :3][:, np.newaxis, :])

    # Torch and Tensorflow's cross product requires both inputs to be of the same size unlike numpy
    # M x 3
    v01 = triangles['face'][:, 1, :3] - triangles['face'][:, 0, :3]
    v12 = triangles['face'][:, 2, :3] - triangles['face'][:, 1, :3]
    v20 = triangles['face'][:, 0, :3] - triangles['face'][:, 2, :3]

    cond_v01 = torch.sum(tensor_cross_prod(v01, v_p0) * normals, dim=-1) >= 0
    cond_v12 = torch.sum(tensor_cross_prod(v12, v_p1) * normals, dim=-1) >= 0
    cond_v20 = torch.sum(tensor_cross_prod(v20, v_p2) * normals, dim=-1) >= 0

    intersection_mask = cond_v01 * cond_v12 * cond_v20
    ray_dist = where(intersection_mask, ray_dist, 1001)

    return {'intersect': intersection_pts, 'normal': result['normal'], 'ray_distance': ray_dist,
            'intersection_mask': intersection_mask}


intersection_fn = {'disk': ray_disk_intersection,
                   'plane': ray_plane_intersection,
                   'sphere': ray_sphere_intersection,
                   'triangle': ray_triangle_intersection,
                   }


def lookat(eye, at, up):
    """Returns a lookat matrix
    :param eye:
    :param at:
    :param up:
    :return:
    """
    if up.size()[-1] == 4:
        assert up.data.numpy()[3] == 0
        up = up[:3]

    if eye.size()[-1] == 4:
        assert abs(eye.data.numpy()[3]) > 0
        eye = eye[:3] / eye[3]

    if at.size()[-1] == 4:
        assert abs(at.data.numpy()[3]) > 0
        at = at[:3] / at[3]

    z = (eye - at)
    z = normalize(z)

    y = normalize(up)
    x = torch.cross(y, z)

    rot_matrix = torch.stack((x, y, z), dim=1).transpose(1, 0)
    rot_translate = torch.cat((rot_matrix, -eye[:3][:, np.newaxis]), dim=1)
    return torch.cat((rot_translate, tch_var_f([0, 0, 0, 1])[np.newaxis, :]), dim=0)


def lookat_inv(eye, at, up):
    """Returns the inverse lookat matrix
    :param eye: camera location
    :param at: lookat point
    :param up: up direction
    :return: 4x4 inverse lookat matrix
    """
    rot_matrix = lookat_rot_inv(eye, at, up)  #torch.stack((x, y, z), dim=1)
    rot_translate = torch.cat((rot_matrix, eye.view(-1, 1)), dim=1)
    return torch.cat((rot_translate, tch_var_f([0, 0, 0, 1.])[np.newaxis, :]), dim=0)


def lookat_rot_inv(eye, at, up):
    """Returns the inverse lookat matrix
    :param eye: camera location
    :param at: lookat point
    :param up: up direction
    :return: 4x4 inverse lookat matrix
    """
    if up.size()[-1] == 4:
        assert up.data.numpy()[3] == 0
        up = up[:3]

    if eye.size()[-1] == 4:
        assert abs(eye.data.numpy()[3]) > 0
        eye = eye[:3] / eye[3]

    if at.size()[-1] == 4:
        assert abs(at.data.numpy()[3]) > 0
        at = at[:3] / at[3]

    z = (eye - at)
    z = normalize(z)

    up = normalize(up)
    x = normalize(torch.cross(up, z))
    # The input `up` vector may not be orthogonal to z.
    y = torch.cross(z, x)

    return torch.stack((x, y, z), dim=1)


def tonemap(im, **kwargs):
    if kwargs['type'] == 'gamma':
        return torch.pow(im, kwargs['gamma'])


def generate_rays(camera):
    viewport = np.array(camera['viewport'])
    W, H = viewport[2] - viewport[0], viewport[3] - viewport[1]
    aspect_ratio = W / H
    fovy = np.array(camera['fovy'])
    focal_length = np.array(camera['focal_length'])
    h = np.tan(fovy / 2) * 2 * focal_length
    w = h * aspect_ratio

    x, y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, H))

    x *= w / 2
    y *= h / 2

    n_pixels = x.size

    x = tch_var_f(x.ravel())
    y = tch_var_f(y.ravel())

    eye = camera['eye'][:3]
    at = camera['at'][:3]
    up = camera['up'][:3]

    ray_dir = torch.stack((x, y, tch_var_f(-np.ones(n_pixels) * focal_length)), dim=0)
    inv_view_matrix = lookat_rot_inv(eye=eye, at=at, up=up)

    ray_dir = torch.mm(inv_view_matrix, ray_dir)

    # normalize ray direction
    ray_dir /= torch.sqrt(torch.sum(ray_dir ** 2, dim=0))

    return eye, ray_dir, H, W


def ray_object_intersections(eye, ray_dir, scene_objects, **kwargs):
    obj_intersections = None
    ray_dist = None
    normals = None
    material_idx = None
    for obj_type in scene_objects:
        result = intersection_fn[obj_type](eye, ray_dir, scene_objects[obj_type], **kwargs)
        curr_intersects = result['intersect']
        curr_ray_dist = result['ray_distance']
        curr_normals = result['normal']
        if curr_ray_dist.dim() == 1:
            curr_ray_dist = curr_ray_dist[np.newaxis, :]
        if curr_intersects.dim() == 1:
            curr_intersects = curr_intersects[np.newaxis, :]
        if curr_normals is not None and curr_normals.dim() == 1:
            curr_normals = curr_normals[np.newaxis, :]

        if obj_intersections is None:
            assert ray_dist is None
            obj_intersections = curr_intersects
            ray_dist = curr_ray_dist
            normals = curr_normals
            material_idx = scene_objects[obj_type]['material_idx']
        else:
            obj_intersections = torch.cat((obj_intersections, curr_intersects), dim=0)
            ray_dist = torch.cat((ray_dist, curr_ray_dist), dim=0)
            if normals is not None:
                normals = torch.cat((normals, curr_normals), dim=0)
            material_idx = torch.cat((material_idx, scene_objects[obj_type]['material_idx']), dim=0)

    return obj_intersections, ray_dist, normals, material_idx


def backface_labeler(eye, scene_objects):
    """Add a binary label per planar geometry.
       0: Facing the camera.
       1: Facing away from the camera, i.e., back-face.
    :param eye: Camera position
    :param scene_objects: Dictionary of scene geometry
    :return: Dictionary of scene geometry with backface label for each geometry
    """
    for obj_type in scene_objects:
        if obj_type == 'sphere':
            continue
        if obj_type == 'triangle':
            pos = scene_objects[obj_type]['face'][:, 0, :3]
        else:
            pos = scene_objects[obj_type]['pos'][:, :3]
        normals = scene_objects[obj_type]['normal'][:, :3]
        cam_dir = normalize(eye[:3] - pos)
        facing_dir = torch.sum(normals * cam_dir, dim=-1)
        scene_objects[obj_type]['facing_dir'] = facing_dir
        scene_objects[obj_type]['backface'] = facing_dir < 0

    return scene_objects
