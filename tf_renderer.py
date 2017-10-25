import numpy as np
import tensorflow as tf


def tensor_cross_prod(u, M):
    """
    :param u:  N x 3
    :param M: N x P x 3
    :return:
    """
    s0 = u[:, 1][:, tf.newaxis] * M[..., 2] - u[:, 2][:, tf.newaxis] * M[..., 1]
    s1 = -u[:, 0][:, tf.newaxis] * M[..., 2] + u[:, 2][:, tf.newaxis] * M[..., 0]
    s2 = u[:, 0][:, tf.newaxis] * M[..., 1] - u[:, 1][:, tf.newaxis] * M[..., 0]

    return tf.stack((s0, s1, s2), axis=2)

def point_along_ray(eye, ray_dir, ray_dist):
    print(ray_dist)
    return eye[tf.newaxis, tf.newaxis, :] + ray_dist[..., tf.newaxis] * ray_dir.T[np.newaxis, ...]


def ray_sphere_intersection(eye, ray_dir, sphere):
    """Bundle of rays intersected with a batch of spheres
    :param eye:
    :param ray_dir:
    :param sphere:
    :return:
    """
    pos = sphere['pos']
    pos_tilde = eye - pos
    radius = sphere['radius']

    a = tf.reduce_sum(ray_dir ** 2, axis=0)
    b = 2 * tf.matmul(pos_tilde, ray_dir)
    c = (tf.reduce_sum(pos_tilde ** 2, axis=1) - radius ** 2)[:, tf.newaxis]

    d_sqr = b ** 2 - 4 * a * c
    intersect_mask = d_sqr >= 0

    # TODO: The following might be more efficient using gather/scatter ?
    d_sqr = tf.where(intersect_mask, d_sqr, tf.zeros_like(d_sqr))

    d = tf.sqrt(d_sqr)
    inv_denom = 1. / (2 * a)

    t1 = (-b - d) * inv_denom
    t2 = (-b + d) * inv_denom

    # get the nearest positive depth
    max_val = tf.maximum(tf.reduce_max(t1), tf.reduce_max(t2)) + 1
    t1 = tf.where(intersect_mask & (t1 >= 0), t1, tf.ones_like(t1) * max_val)
    t2 = tf.where(intersect_mask & (t2 >= 0), t2, tf.ones_like(t2) * max_val)

    ray_dist = tf.reduce_min(tf.stack((t1, t2), axis=2), axis=2)
    ray_dist = tf.where(intersect_mask, ray_dist, tf.zeros_like(ray_dist))

    intersection_pts = point_along_ray(eye, ray_dir, ray_dist)

    normals = intersection_pts - pos[:, np.newaxis, :]
    normals /= tf.sqrt(tf.reduce_sum(normals ** 2, axis=-1))[..., tf.newaxis]

    # # Index for gather/scatter
    # # Assuming that this function is called with the number of objects known at the graph construction time.
    # M, N, D = np.mgrid[0:intersect_mask.shape[0], 0:intersect_mask[1], 0:4]
    normals = tf.where(tf.tile(intersect_mask[..., tf.newaxis], (1, 1, 4)), normals, tf.zeros_like(normals))

    return {'intersect': intersection_pts, 'normal': normals, 'ray_distance': ray_dist,
            'intersection_mask': intersect_mask}


def ray_plane_intersection(eye, ray_dir, plane):
    """Intersection a bundle of rays with a batch of planes
    :param eye: Camera's center of projection
    :param ray_dir: Ray direction
    :param plane: Plane specification
    :return:
    """
    pos = plane['pos']
    normal = plane['normal']
    dist = tf.reduce_sum(pos * normal, axis=1)

    denom = tf.matmul(normal, ray_dir)

    intersection_mask = tf.abs(denom) > 0
    denom = tf.where(intersection_mask, denom, tf.ones_like(denom))
    print(dist)
    ray_dist = (dist[:, tf.newaxis] - tf.matmul(normal, eye[:, tf.newaxis])) / denom
    ray_dist = tf.where(intersection_mask, ray_dist, tf.zeros_like(ray_dist))

    intersection_pts = point_along_ray(eye, ray_dir, ray_dist)
    normals = tf.ones_like(intersection_pts) * normal[:, tf.newaxis, :]
    normals = tf.where(tf.tile(intersection_mask[..., tf.newaxis], (1, 1, 4)), normals, tf.zeros_like(normals))

    return {'intersect': intersection_pts, 'normal': normals, 'ray_distance': ray_dist,
            'intersection_mask': intersection_mask}


def ray_disk_intersection(eye, ray_dir, disks):
    result = ray_plane_intersection(eye, ray_dir, disks)
    intersection_pts = result['intersect']
    normals = result['normal']
    ray_dist = result['ray_distance']

    centers = disks['pos']

    dist_sqr = tf.reduce_sum((intersection_pts - centers[:, tf.newaxis, :]) ** 2, axis=-1)

    # Intersection mask
    intersection_mask = (dist_sqr <= disks['radius'][:, tf.newaxis] ** 2)
    ray_dist = tf.where(intersection_mask, ray_dist, tf.zeros_like(ray_dist))
    #intersection_pts[~mask_intersect] = np.inf
    #ray_dist[~mask_intersect] = np.inf
    normals = tf.where(tf.tile(intersection_mask[..., tf.newaxis], (1, 1, 4)), normals, tf.zeros_like(normals))

    return {'intersect': intersection_pts, 'normal': normals, 'ray_distance': ray_dist,
            'intersection_mask': intersection_mask}


def ray_triangle_intersection(eye, ray_dir, triangles):
    """Intersection of a bundle of rays with a batch of triangles.
    Assumes that the triangles vertices are specified as F x 3 x 4 matrix where F is the number of faces and
    the normals for all faces are precomputed and in a matrix of size F x 4 (i.e., similar to the normals for other
    geometric primitives). Note that here the number of faces F is the same as number of primitives M.
    :param eye:
    :param ray_dir:
    :param triangles:
    :return:
    """

    planes = {'pos': triangles['faces'][:, 0, :], 'normal': triangles['normal']}
    print(planes)
    result = ray_plane_intersection(eye, ray_dir, planes)
    intersection_pts = result['intersect']  # M x N x 4 matrix where M is the number of objects and N pixels.
    normals = result['normal'][..., :3]  # M x N x 4
    ray_dist = result['ray_distance']

    num_pixels = intersection_pts.shape[1]
    # check if intersection point is inside or outside the triangle
    # M x N x 3
    v_p0 = (intersection_pts - triangles['faces'][:, 0, :][:, tf.newaxis, :])[..., :3]
    v_p1 = (intersection_pts - triangles['faces'][:, 1, :][:, tf.newaxis, :])[..., :3]
    v_p2 = (intersection_pts - triangles['faces'][:, 2, :][:, tf.newaxis, :])[..., :3]

    print(v_p0)
    print('sub', tf.constant(triangles['faces'][:, 1, :3] - triangles['faces'][:, 0, :3])[:, tf.newaxis, :])
    # Tensorflow's cross product requires both inputs to be of the same size unlike numpy
    # M x 3
    v01 = triangles['faces'][:, 1, :3] - triangles['faces'][:, 0, :3]
    v12 = triangles['faces'][:, 2, :3] - triangles['faces'][:, 1, :3]
    v20 = triangles['faces'][:, 0, :3] - triangles['faces'][:, 2, :3]

    print(v01.shape)
    # cond_v01 = []
    # cond_v12 = []
    # cond_v20 = []
    # for col_idx in range(num_pixels):
    #     cond_v01.append(tf.reduce_sum(tf.cross(v01, v_p0[:, col_idx, :]) * normals[:, col_idx, :], axis=-1) >= 0)
    #print(cond_v01)
    #assert False
    cond_v01 = tf.reduce_sum(tensor_cross_prod(v01, v_p0) * normals, axis=-1) >= 0
    cond_v12 = tf.reduce_sum(tensor_cross_prod(v12, v_p1) * normals, axis=-1) >= 0
    cond_v20 = tf.reduce_sum(tensor_cross_prod(v20, v_p2) * normals, axis=-1) >= 0

    intersection_mask = cond_v01 & cond_v12 & cond_v20
    ray_dist = tf.where(intersection_mask, ray_dist, tf.zeros_like(ray_dist))
    #ray_dist[~intersection_mask] = np.inf

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
    matrix[:3, 3] = -eye[:3] / eye[3]
    return matrix


def lookat_inv(eye, at, up):
    """Returns the inverse lookat matrix
    :param eye: camera location
    :param at: lookat point
    :param up: up direction
    :return: 4x4 inverse lookat matrix
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
    matrix[:3, :3] = np.stack((x, y, z), axis=1)
    matrix[:3, 3] = eye[:3] / eye[3]
    return matrix


def tonemap(im, **kwargs):
    if kwargs['type'] == 'gamma':
        return im ** kwargs['gamma']


def generate_rays(camera):
    viewport = camera['viewport']
    W, H = viewport[2] - viewport[0], viewport[3] - viewport[1]
    aspect_ratio = W / float(H)

    fovy = camera['fovy']
    focal_length = camera['focal_length']
    h = np.tan(fovy / 2) * 2 * focal_length
    w = h * aspect_ratio
    x, y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, H))

    x *= w / 2
    y *= h / 2

    eye = np.array(camera['eye'])
    ray_dir = np.stack((x.ravel(), y.ravel(), -np.ones(x.size) * focal_length, np.zeros(x.size)), axis=0)
    # view_matrix = lookat(eye=eye, at=camera['at'], up=camera['up'])
    inv_view_matrix = lookat_inv(eye=eye, at=camera['at'], up=camera['up'])
    print(inv_view_matrix, np.linalg.inv(inv_view_matrix))
    ray_dir = np.dot(inv_view_matrix, ray_dir)

    # normalize ray direction
    ray_dir /= np.sqrt(np.sum(ray_dir ** 2, axis=0))

    return eye, ray_dir, H, W


def ray_object_intersections(eye, ray_dir, scene_objects):
    obj_intersections = None
    ray_dist = None
    normals = None
    material_idx = None
    for obj_type in scene_objects:
        result = intersection_fn[obj_type](eye, ray_dir, scene_objects[obj_type])
        curr_intersects = result['intersect']
        curr_ray_dist = result['ray_distance']
        curr_normals = result['normal']
        if len(curr_ray_dist.shape) == 1:
            curr_ray_dist = curr_ray_dist[tf.newaxis, :]
        if len(curr_intersects.shape) == 1:
            curr_intersects = curr_intersects[tf.newaxis, :]
        if len(curr_normals.shape) == 1:
            curr_normals = curr_normals[tf.newaxis, :]

        if obj_intersections is None:
            assert ray_dist is None
            obj_intersections = curr_intersects
            ray_dist = curr_ray_dist
            normals = curr_normals
            material_idx = scene_objects[obj_type]['material_idx']
        else:
            obj_intersections = tf.concat((obj_intersections, curr_intersects), axis=0)
            ray_dist = tf.concat((ray_dist, curr_ray_dist), axis=0)
            normals = tf.concat((normals, curr_normals), axis=0)
            material_idx = tf.concat((material_idx, scene_objects[obj_type]['material_idx']), axis=0)

    return obj_intersections, ray_dist, normals, material_idx


def render(scene):
    """
    :param scene: Scene description
    :return: [H, W, 3] image
    """
    # Construct rays from the camera's eye position through the screen coordinates
    camera = scene['camera']
    eye, ray_dir, H, W = generate_rays(camera)

    # Ray-object intersections
    scene_objects = scene['objects']
    obj_intersections, ray_dist, normals, material_idx = ray_object_intersections(eye, ray_dir, scene_objects)

    # Valid distances
    pixel_dist = ray_dist
    valid_pixels = (camera['near'] <= ray_dist) & (ray_dist <= camera['far'])
    pixel_dist = tf.where(valid_pixels, ray_dist, tf.ones_like(ray_dist) * (camera['far'] + 1))

    # Nearest object needs to be compared for valid regions only
    nearest_obj = tf.argmin(pixel_dist, axis=0)
    C = np.arange(0, H * W)  # pixel idx

    # Create depth image for visualization
    # use nearest_obj for gather/select the pixel color
    print(pixel_dist, nearest_obj)
    # Gather the pixel values
    nearest_idx = tf.stack((nearest_obj, C), axis=1)
    print(nearest_idx)
    im_depth = tf.reshape(tf.gather_nd(pixel_dist, nearest_idx), (H, W))

    ##############################
    # Fragment processing
    ##############################
    # Lighting
    color_table = scene['colors']
    light_pos = scene['lights']['pos']
    light_clr_idx = scene['lights']['color_idx']
    light_colors = color_table[light_clr_idx]

    # Generate the fragments
    """
    Get the normal and material for the visible objects.
    """
    materials = tf.Variable(scene['materials']['albedo'], trainable=scene['materials']['trainable'])
    frag_normals = tf.gather_nd(normals, nearest_idx)
    frag_pos = tf.gather_nd(obj_intersections, nearest_idx)
    tmp_idx = tf.gather(material_idx, nearest_obj)
    frag_albedo = tf.gather(materials, tmp_idx)

    # Fragment shading
    light_dir = light_pos[np.newaxis, :] - frag_pos[:, tf.newaxis, :]
    light_dir_norm = tf.sqrt(tf.reduce_sum(light_dir ** 2, axis=-1))[..., tf.newaxis]
    #light_dir_norm[light_dir_norm <= 0 | np.isinf(light_dir_norm)] = 1
    light_dir /= light_dir_norm

    # return im_depth, nearest_obj, normals, obj_intersections, material_idx

    im_color = tf.reduce_sum(frag_normals[:, tf.newaxis, :] * light_dir, axis=-1)[..., tf.newaxis] * \
               light_colors[tf.newaxis, ...] * frag_albedo[:, tf.newaxis, :]

    im = tf.reshape(tf.reduce_sum(im_color, axis=1), (H, W, 3))

    # only consider pixels within valid range and non-negative
    valid_idx = tf.where((camera['near'] <= im_depth) & (im_depth <= camera['far']))
    im = tf.scatter_nd(valid_idx, tf.gather_nd(im, valid_idx), im.shape)

    # clip negative
    im = tf.where(im >= 0, im, tf.zeros_like(im))

    # Tonemapping
    if 'tonemap' in scene:
        im = tonemap(im, **scene['tonemap'])

    return {'image': im,
            'depth': im_depth,
            'ray_dist': ray_dist,
            'obj_dist': pixel_dist,
            'nearest': tf.reshape(nearest_obj, (H, W)),
            'ray_dir': ray_dir,
            'valid_pixels': valid_pixels,
            'materials': materials
            }


################
def optimize_scene(out_dir, max_iter=1000, learning_rate=1e-3, print_interval=10, imsave_interval=10,
                   b_optimize=False):
    import os
    from scipy.misc import imsave
    import matplotlib.pyplot as plt


    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

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
                       # Light attenuation factors have the form (kc, kl, kq) and eq: 1/(kc + kl * d + kq * d^2)
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

    # Solve a toy optimization problem
    # create a target image

    graph = tf.Graph()
    with graph.as_default():
        res = render(scene_basic)

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        im, im_depth_, nearest, materials_target = sess.run([res['image'], res['depth'], res['nearest'], res['materials']])
    im_target = im
    if b_optimize:
        imsave(out_dir + '/target.png', im_target)
    del graph

    # optimize
    loss_per_iter = []
    materials_final = None
    if b_optimize:
        scene_test = scene_basic
        scene_test['materials']['albedo'][3] = np.array([0.1, 0.8, 0.9])
        scene_test['materials']['albedo'][4] = np.array([0.1, 0.8, 0.9])
        scene_test['materials']['albedo'][5] = np.array([0.9, 0.1, 0.1])

        graph = tf.Graph()
        with graph.as_default():
            res = render(scene_test)
            loss = tf.reduce_mean((res['image'] - im_target) ** 2)
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


        with tf.Session(graph=graph) as sess:
            tf.global_variables_initializer().run()
            h = plt.figure()
            for iter in range(max_iter + 1):
                if iter > 0:
                    sess.run(opt)
                loss_ = sess.run(loss)
                loss_per_iter.append(loss_)
                if iter % print_interval == 0 or iter == max_iter:
                    print('%d. Loss: %.6f' % (iter, loss_))
                if iter % imsave_interval == 0 or iter == max_iter:
                    im_optimized = sess.run(res['image'])
                    out_filename = out_dir + '/im_{:05d}.png'.format(iter)
                    imsave(out_filename, im_optimized)
                    plt.figure(h.number)
                    plt.clf()
                    plt.imshow(im_optimized)
                    plt.axis('off')
                    plt.title('Iteration %d, MSE Loss: %.06f' % (iter, loss_))
                    plt.savefig(out_dir + '/fig_{:05d}.png'.format(iter))

            materials_final = sess.run(res['materials'])

    plt.ion()
    plt.figure()
    plt.imshow(im_target)
    plt.title('Target Final Rendered Image')

    if b_optimize:
        plt.figure()
        plt.imshow(im)
        plt.title('Final Rendered Image')

    # for better contrast in the matplotlib rendering set the far distance to inf
    im_depth_[im_depth_ >= scene_basic['camera']['far']] = np.inf
    plt.figure()
    plt.imshow(im_depth_)
    plt.title('Depth Image')
    plt.colorbar()
    plt.savefig(out_dir + '/DepthImage.png')

    if b_optimize:
        plt.figure()
        plt.plot(loss_per_iter)
        plt.xlabel('Iterations', fontsize=14)
        plt.title('MSE Loss', fontsize=14)
        plt.grid(True)
        plt.savefig(out_dir + '/loss.png')

    plt.show()

    return materials_final, materials_target, loss_per_iter


if __name__ == '__main__':
    mat_final, mat_target, loss_per_iter = optimize_scene(out_dir='./opt_res_2', max_iter=2000,
                                                          imsave_interval=20, print_interval=20,
                                                          b_optimize=True)
    print('final', mat_final)
    print('target', mat_target)
