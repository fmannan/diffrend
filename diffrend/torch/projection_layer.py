import torch
import numpy as np
from diffrend.torch.utils import cam_to_world, world_to_cam, world_to_cam_batched, cam_to_world_batched, get_data
from diffrend.torch.render import render_scene, load_scene, make_torch_var
from diffrend.torch.utils import make_list2np, tch_var_f, tch_var_l, scatter_mean_dim0
from diffrend.torch.renderer import z_to_pcl_CC, z_to_pcl_CC_batched
import copy


"""Projection Layer
1. Project to the image plane
2. Transform to viewport
3. Find pixel index
4. scatter_mean surfel data to pixels
"""


def project_surfels(surfel_pos_WC, camera):
    """Returns the surfel coordinates (defined in world coordinate) on a projection plane
    (in the camera's coordinate frame)

    Args:
        surfels: [batch_size, num_surfels, 3D pos]
        camera: [{'eye': [num_batches,...], 'lookat': [num_batches,...], 'up': [num_batches,...],
                    'viewport': [0, 0, W, H], 'fovy': <radians>}]

    Returns:
        [batch_size, num_surfels, 2D pos]

    """
    surfels_cc = world_to_cam_batched(surfel_pos_WC, None, camera)['pos']

    """ If a 3D point (X, Y, Z) is projected to a 2D point (x, y) then for focal_length f,
    x/f = X/Z and y/f = Y/Z
    """
    focal_length = camera['focal_length']
    x = focal_length * surfels_cc[..., 0] / surfels_cc[..., 2]
    y = focal_length * surfels_cc[..., 1] / surfels_cc[..., 2]

    return torch.stack((x, y), dim=-1)


def project_image_coordinates(surfels, camera):
    """Project surfels given in world coordinate to the camera's projection plane.

    Args:
        surfels: [batch_size, pos]
        camera: [{'eye': [num_batches,...], 'lookat': [num_batches,...], 'up': [num_batches,...],
                    'viewport': [0, 0, W, H], 'fovy': <radians>}]

    Returns:
        RGB image of dimensions [batch_size, H*W, 3] from projected surfels
        Note that the range of possible coordinates is restricted to be between 0
        and W*H (inclusive). This is inclusive because we use the last index as
        a "dump" for any index that falls outside of the camera's field of view
    """
    surfels_2d = project_surfels(surfels, camera)

    # Rasterize
    viewport = make_list2np(camera['viewport'])
    W, H = float(viewport[2] - viewport[0]), float(viewport[3] - viewport[1])
    aspect_ratio = float(W) / float(H)

    fovy = make_list2np(camera['fovy'])
    focal_length = make_list2np(camera['focal_length'])
    h = np.tan(fovy / 2) * 2 * focal_length
    w = h * aspect_ratio

    px_coord = surfels_2d * tch_var_f([-(W - 1) / w, (H - 1) / h]).unsqueeze(-2) + tch_var_f([W / 2., H / 2.]).unsqueeze(-2)
    px_coord_idx = torch.round(px_coord - 0.5).long()

    px_idx = px_coord_idx[..., 1] * W + px_coord_idx[..., 0]

    max_idx = W * H # Index used if the indices are out of bounds of the camera
    max_idx_tensor = tch_var_l([max_idx])

    # Map out of bounds pixels to the last (extra) index
    mask = (px_coord_idx[..., 1] < 0) | (px_coord_idx[..., 0] < 0) | (px_coord_idx[..., 1] >= H) | (px_coord_idx[..., 1] >= W)
    px_idx = torch.where(mask, max_idx_tensor, px_idx)

    return px_idx, px_coord


def projection_renderer(surfels, rgb, camera):
    """Project surfels given in world coordinate to the camera's projection plane.

    Args:
        surfels: [batch_size, num_surfels, pos]
        rgb: [batch_size, num_surfels, D-channel data] or [batch_size, H, W, D-channel data]
        camera: [{'eye': [num_batches,...], 'lookat': [num_batches,...], 'up': [num_batches,...],
                    'viewport': [0, 0, W, H], 'fovy': <radians>}]

    Returns:
        RGB image of dimensions [batch_size, H, W, 3] from projected surfels

    """
    px_idx, px_coord = project_image_coordinates(surfels, camera)
    viewport = make_list2np(camera['viewport'])
    W = int(viewport[2] - viewport[0])
    rgb_reshaped = rgb.view(rgb.size(0), -1, rgb.size(-1))
    rgb_out, mask = scatter_mean_dim0(rgb_reshaped, px_idx.long())
    return rgb_out.reshape(rgb.shape), mask.reshape(rgb.shape)


def randomly_rotate_cameras(camera, theta_range=[-np.pi / 2, np.pi / 2], phi_range=[-np.pi, np.pi]):
    """
    Given a batch of camera positions, rotate the 'eye' properties around the 'lookat' position
    :param camera: [{'eye': [num_batches,...], 'lookat': [num_batches,...], 'up': [num_batches,...],
                    'viewport': [0, 0, W, H], 'fovy': <radians>}]
    
    Modifies the camera object in place
    """
    # Sample a theta and phi to add to the current camera rotation
    theta_samples, phi_samples = uniform_sample_sphere_patch(get_data(camera['eye']).shape[0], theta_range, phi_range)

    # Get the current camera rotation (relative to the 'lookat' position)
    camera_eye = cartesian_to_spherical(get_data(camera['eye']) - get_data(camera['at']))

    # Rotate the camera
    new_thetas = camera_eye[...,0] + theta_samples
    new_phis = camera_eye[...,1] + phi_samples

    # Go back to cartesian coordinates and place the camera back relative to the 'lookat' position
    camera_eye = spherical_to_cartesian(new_thetas, new_phis, radius=np.expand_dims(camera_eye[...,2], -1))

    if camera['at'].shape[-1] == 4:
        zeros = np.zeros((camera_eye.shape[0], 1))
        camera_eye = np.concatenate((camera_eye, zeros), axis=-1)

    camera['eye'] = tch_var_f(camera_eye) + camera['at']


def uniform_sample_sphere_patch(num_samples, theta_range=[0, np.pi], phi_range=[0, 2 * np.pi]):
    """
       Generate uniform random samples a patch defined by theta and phi ranges
       on the surface of the sphere, around the z-axis.
       theta_range: angle from the z-axis
       phi_range: range of angles on the xy plane from the x-axis

       Returns: (num_samples) for theta, and (num_samples) for phi angles of the sampled camera position
    """
    pts_2d = np.random.rand(num_samples, 2)
    s_range = 1 - np.cos(np.array(theta_range) / 2) ** 2
    t_range = np.array(phi_range) / (2 * np.pi)
    s = min(s_range) + pts_2d[:, 0] * (max(s_range) - min(s_range))
    t = min(t_range) + pts_2d[:, 1] * (max(t_range) - min(t_range))
    # theta is angle from the z-axis
    theta = 2 * np.arccos(np.sqrt(1 - s))
    phi = 2 * np.pi * t
    return theta, phi


def spherical_to_cartesian(theta, phi, radius=1):
    return np.stack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                     np.cos(theta)), axis=-1) * radius


def cartesian_to_spherical(coords):
    x, y, z = coords[...,0], coords[...,1], coords[...,2]
    norm2 = x**2 + y**2
    return np.stack((np.arctan2(np.sqrt(norm2), z), np.arctan2(y, x), np.sqrt(norm2 + z**2)), axis=-1)


def test_raster_coordinates(scene, batch_size):
    """Test if the projected raster coordinates are correct

    Args:
        scene: Path to scene file

    Returns:
        None

    """
    res = render_scene(scene)
    scene = make_torch_var(load_scene(scene))
    pos_cc = res['pos'].reshape(1, -1, res['pos'].shape[-1])
    pos_cc = pos_cc.repeat(batch_size, 1, 1)

    camera = scene['camera']
    camera['eye'] = camera['eye'].repeat(batch_size, 1)
    camera['at'] = camera['at'].repeat(batch_size, 1)
    camera['up'] = camera['up'].repeat(batch_size, 1)

    viewport = make_list2np(camera['viewport'])
    W, H = float(viewport[2] - viewport[0]), float(viewport[3] - viewport[1])
    px_coord_idx, px_coord = project_image_coordinates(pos_cc, camera)
    xp, yp = np.meshgrid(np.linspace(0, W - 1, int(W)), np.linspace(0, H - 1, int(H)))
    xp = xp.ravel()[None, ...].repeat(batch_size, axis=0)
    yp = yp.ravel()[None, ...].repeat(batch_size, axis=0)

    px_coord = torch.round(px_coord - 0.5).long()

    np.testing.assert_array_almost_equal(xp, get_data(px_coord[..., 0]))
    np.testing.assert_array_almost_equal(yp, get_data(px_coord[..., 1]))


def test_render_projection_consistency(scene, batch_size):
    """ First render using the full renderer to get the surfel position and color
    and then render using the projection layer for testing

    Returns:

    """
    res = render_scene(scene)

    scene = make_torch_var(load_scene(scene))
    pos_cc = res['pos'].reshape(-1, res['pos'].shape[-1]).repeat(batch_size, 1, 1)

    camera = scene['camera']
    camera['eye'] = camera['eye'].repeat(batch_size, 1)
    camera['at'] = camera['at'].repeat(batch_size, 1)
    camera['up'] = camera['up'].repeat(batch_size, 1)

    image = res['image'].repeat(batch_size, 1, 1, 1)

    im, mask = projection_renderer(pos_cc, image, camera)
    diff = np.abs(get_data(image) - get_data(im))
    np.testing.assert_(diff.sum() < 1e-10, 'Non-zero difference.')


def test_visual_render(scene, batch_size):
    """
    Test that outputs visual images for the user to compare
    """
    from torchvision.utils import save_image

    res = render_scene(scene)

    scene = make_torch_var(load_scene(scene))

    camera = scene['camera']
    camera['eye'] = camera['eye'].repeat(batch_size, 1)
    camera['at'] = camera['at'].repeat(batch_size, 1)
    camera['up'] = camera['up'].repeat(batch_size, 1)

    depth = res['depth'].repeat(batch_size, 1, 1).reshape(batch_size, -1)
    pos_cc = z_to_pcl_CC_batched(-depth, scene['camera']) # NOTE: z = -depth
    pos_wc = cam_to_world_batched(pos_cc, None, scene['camera'])['pos']

    randomly_rotate_cameras(camera, theta_range=[-np.pi / 16, np.pi / 16], phi_range=[-np.pi / 8, np.pi / 8])

    image = res['image'].repeat(batch_size, 1, 1, 1)
    save_image(image.permute(0, 3, 1, 2), 'test-original.png', nrow=2)

    im, mask = projection_renderer(pos_wc, image, camera)
    save_image(im.permute(0, 3, 1, 2), 'test-rotated-reprojected.png', nrow=2)


def test_transformation_consistency(scene, batch_size):
    print('test_transformation_consistency')
    res = render_scene(scene)
    scene = make_torch_var(load_scene(scene))
    pos_cc = res['pos'].reshape(-1, res['pos'].shape[-1])
    normal_cc = res['normal'].reshape(-1, res['normal'].shape[-1])
    surfels = cam_to_world(pos_cc,
                           normal_cc,
                           scene['camera'])
    surfels_cc = world_to_cam(surfels['pos'], surfels['normal'], scene['camera'])

    np.testing.assert_array_almost_equal(get_data(pos_cc), get_data(surfels_cc['pos'][:, :3]))
    np.testing.assert_array_almost_equal(get_data(normal_cc), get_data(surfels_cc['normal'][:, :3]))


def test_optimization(scene, batch_size, print_interval=20, imsave_interval=20, max_iter=100,
                      out_dir='./proj_tmp/'):
    """ First render using the full renderer to get the surfel position and color
    and then render using the projection layer for testing

    Returns:

    """
    from torch import optim
    import os
    import matplotlib.pyplot as plt
    plt.ion()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    res = render_scene(scene)

    scene = make_torch_var(load_scene(scene))
    pos_wc = res['pos'].reshape(-1, res['pos'].shape[-1]).repeat(batch_size, 1, 1)

    camera = scene['camera']
    camera['eye'] = camera['eye'].repeat(batch_size, 1)
    camera['at'] = camera['at'].repeat(batch_size, 1)
    camera['up'] = camera['up'].repeat(batch_size, 1)

    target_image = res['image'].repeat(batch_size, 1, 1, 1)

    input_image = target_image + 0.1 * torch.randn(target_image.size(), device=target_image.device)
    input_image.requires_grad = True

    criterion = torch.nn.MSELoss(size_average=True).cuda()
    optimizer = optim.Adam([input_image], lr=1e-2)

    h1 = plt.figure()
    loss_per_iter = []
    for iter in range(100):
        im_est, mask = projection_renderer(pos_wc, input_image, camera)
        optimizer.zero_grad()
        loss = criterion(im_est * 255, target_image * 255)
        loss_ = get_data(loss)
        loss_per_iter.append(loss_)
        if iter % print_interval == 0 or iter == max_iter - 1:
            print('{}. Loss: {}'.format(iter, loss_))
        if iter % imsave_interval == 0 or iter == max_iter - 1:
            im_out_ = get_data(input_image)
            im_out_ = np.uint8(255 * im_out_ / im_out_.max())
            plt.figure(h1.number)
            plt.imshow(im_out_[0].squeeze())
            plt.title('%d. loss= %f' % (iter, loss_))
            plt.savefig(out_dir + '/fig_%05d.png' % iter)

        loss.backward()
        optimizer.step()

def test_depth_optimization(scene, batch_size, print_interval=20, imsave_interval=20, max_iter=100,
                      out_dir='./proj_tmp_depth/'):
    """ First render using the full renderer to get the surfel position and color
    and then render using the projection layer for testing

    Returns:

    """
    from torch import optim
    import os
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import imageio
    plt.ion()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    res = render_scene(scene)

    scene = make_torch_var(load_scene(scene))
    # true_pos_wc = res['pos'].reshape(-1, res['pos'].shape[-1]).repeat(batch_size, 1, 1)
    true_input_img = res['image'].unsqueeze(0).repeat(batch_size, 1, 1, 1)

    camera = scene['camera']
    camera['eye'] = camera['eye'].repeat(batch_size, 1)
    camera['at'] = camera['at'].repeat(batch_size, 1)
    camera['up'] = camera['up'].repeat(batch_size, 1)

    true_depth = res['depth'].repeat(batch_size, 1, 1).reshape(batch_size, -1)
    depth = true_depth.clone() + 0.1 * torch.randn(res['depth'].size(), device=res['depth'].device)
    depth.requires_grad = True

    true_pos_cc = z_to_pcl_CC_batched(-true_depth, camera) # NOTE: z = -depth
    true_pos_wc = cam_to_world_batched(true_pos_cc, None, camera)['pos']

    rotated_camera = copy.deepcopy(camera)
    randomly_rotate_cameras(rotated_camera, theta_range=[-np.pi / 16, np.pi / 16], phi_range=[-np.pi / 8, np.pi / 8])
    target_image, _ = projection_renderer(true_pos_wc, true_input_img, rotated_camera)

    input_image = true_input_img # + 0.1 * torch.randn(target_image.size(), device=target_image.device)

    criterion = torch.nn.MSELoss(size_average=True).cuda()
    optimizer = optim.Adam([depth], lr=1e-2)

    h1 = plt.figure()
    fig_imgs = []
    depth_imgs = []
    out_imgs = []

    imageio.imsave(out_dir + '/optimization_target_image.png', target_image[0])

    loss_per_iter = []
    for iter in range(100):
        pos_cc = z_to_pcl_CC_batched(-depth, camera) # NOTE: z = -depth
        pos_wc = cam_to_world_batched(pos_cc, None, camera)['pos']
        im_est, mask = projection_renderer(pos_wc, input_image, rotated_camera)
        optimizer.zero_grad()
        loss = criterion(im_est * 255, target_image * 255)
        loss_ = get_data(loss)
        loss_per_iter.append(loss_)
        if iter % print_interval == 0 or iter == max_iter - 1:
            print('{}. Loss: {}'.format(iter, loss_))
        if iter % imsave_interval == 0 or iter == max_iter - 1:
            # Input image
            im_out_ = get_data(input_image.detach())
            im_out_ = np.uint8(255 * im_out_ / im_out_.max())
            fig = plt.figure(h1.number)
            plot = fig.add_subplot(111)
            plot.imshow(im_out_[0].squeeze())
            plot.set_title('%d. loss= %f' % (iter, loss_))
            # plt.savefig(out_dir + '/fig_%05d.png' % iter)
            fig_data = np.array(fig.canvas.renderer._renderer)
            fig_imgs.append(fig_data)

            # Depth
            im_out_ = get_data(depth.view(*input_image.size()[:-1], 1).detach())
            im_out_ = np.uint8(255 * im_out_ / im_out_.max())
            fig = plt.figure(h1.number)
            plot = fig.add_subplot(111)
            plot.imshow(im_out_[0].squeeze())
            plot.set_title('%d. loss= %f' % (iter, loss_))
            # plt.savefig(out_dir + '/fig_%05d.png' % iter)
            depth_data = np.array(fig.canvas.renderer._renderer)
            depth_imgs.append(depth_data)

            # Output image
            im_out_ = get_data(im_est.detach())
            im_out_ = np.uint8(255 * im_out_ / im_out_.max())
            fig = plt.figure(h1.number)
            plot = fig.add_subplot(111)
            plot.imshow(im_out_[0].squeeze())
            plot.set_title('%d. loss= %f' % (iter, loss_))
            # plt.savefig(out_dir + '/fig_%05d.png' % iter)
            out_data = np.array(fig.canvas.renderer._renderer)
            out_imgs.append(out_data)

        # loss.backward()
        # optimizer.step()
    
    imageio.mimsave(out_dir + '/optimization_anim_in.gif', fig_imgs)
    imageio.mimsave(out_dir + '/optimization_anim_depth.gif', depth_imgs)
    imageio.mimsave(out_dir + '/optimization_anim_out.gif', out_imgs)


def test_depth_to_world_consistency(scene, batch_size):
    res = render_scene(scene)

    scene = make_torch_var(load_scene(scene))

    pos_wc1 = res['pos'].reshape(-1, res['pos'].shape[-1])

    # First test the non-batched z_to_pcl_CC method:
    depth = res['depth'].reshape(-1)
    pos_cc2 = z_to_pcl_CC(-depth, scene['camera']) # NOTE: z = -depth
    pos_wc2 = cam_to_world(pos_cc2, None, scene['camera'])['pos']

    np.testing.assert_array_almost_equal(pos_wc1, pos_wc1)

    # Then test the batched version:
    camera = scene['camera']
    camera['eye'] = camera['eye'].repeat(batch_size, 1)
    camera['at'] = camera['at'].repeat(batch_size, 1)
    camera['up'] = camera['up'].repeat(batch_size, 1)

    pos_wc1 = pos_wc1.repeat(batch_size, 1, 1)

    depth = res['depth'].repeat(batch_size, 1, 1).reshape(batch_size, -1)
    pos_cc2 = z_to_pcl_CC_batched(-depth, camera) # NOTE: z = -depth
    pos_wc2 = cam_to_world_batched(pos_cc2, None, camera)['pos']

    np.testing.assert_array_almost_equal(pos_wc1[...,:3], pos_wc2[...,:3])


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(usage="render.py --scene scene_filename --out_dir output_dir")
    parser.add_argument('--scene', type=str, default='../../scenes/halfbox_sphere_cube.json', help='Path to the model file')
    parser.add_argument('--out_dir', type=str, default='./render_samples/', help='Directory for rendered images.')
    args = parser.parse_args()
    print(args)
    assert os.path.exists(args.scene), print('{} not found'.format(args.scene))

    scene = args.scene
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    batch_size = 6

    test_depth_to_world_consistency(scene, batch_size)
    test_visual_render(scene, batch_size)
    test_render_projection_consistency(scene, batch_size)
    test_raster_coordinates(scene, batch_size)
    test_transformation_consistency(scene, batch_size)
    test_optimization(scene, 1)
    test_depth_optimization(scene, 1)


if __name__ == '__main__':
    main()
