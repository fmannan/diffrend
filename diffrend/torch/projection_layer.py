import torch
import numpy as np
from diffrend.torch.utils import cam_to_world, world_to_cam, world_to_cam_batched, cam_to_world_batched, get_data
from diffrend.torch.render import render_scene, load_scene, make_torch_var
from diffrend.torch.utils import make_list2np, tch_var_f, tch_var_l, scatter_mean_dim0, scatter_weighted_blended_oit, nonzero_divide
from diffrend.torch.renderer import z_to_pcl_CC, z_to_pcl_CC_batched, render
import copy
import math


"""Projection Layer
1. Project to the image plane
2. Transform to viewport
3. Find pixel index
4. scatter_mean surfel data to pixels
"""


def project_surfels(surfel_pos_WC, camera):
    """Returns the surfel coordinates (defined in world coordinate) on a projection plane
    (in the camera's coordinate frame), including the depth (ray distance)

    Args:
        surfels: [batch_size, num_surfels, 3D pos]
        camera: [{'eye': [num_batches,...], 'lookat': [num_batches,...], 'up': [num_batches,...],
                    'viewport': [0, 0, W, H], 'fovy': <radians>}]

    Returns:
        [batch_size, num_surfels, 3D pos]

    """
    surfels_cc = world_to_cam_batched(surfel_pos_WC, None, camera)['pos']

    """ If a 3D point (X, Y, Z) is projected to a 2D point (x, y) then for focal_length f,
    x/f = X/Z and y/f = Y/Z
    """
    focal_length = camera['focal_length']
    x = focal_length * nonzero_divide(surfels_cc[..., 0], surfels_cc[..., 2])
    y = focal_length * nonzero_divide(surfels_cc[..., 1], surfels_cc[..., 2])
    z = -surfels_cc[..., 2] # use depth perpendicular to cam plane so that it matches generated depth from NN

    return torch.stack((x, y, z), dim=-1)


def project_image_coordinates(surfels, camera):
    """Project surfels given in world coordinate to the camera's projection plane.

    Args:
        surfels: [batch_size, pos]
        camera: [{'eye': [num_batches,...], 'lookat': [num_batches,...], 'up': [num_batches,...],
                    'viewport': [0, 0, W, H], 'fovy': <radians>}]

    Returns:
        Image of destination indices of dimensions [batch_size, H*W]
        Note that the range of possible coordinates is restricted to be between 0
        and W*H (inclusive). This is inclusive because we use the last index as
        a "dump" for any index that falls outside of the camera's field of view
    """
    surfels_plane = project_surfels(surfels, camera)

    # Rasterize
    viewport = make_list2np(camera['viewport'])
    W, H = float(viewport[2] - viewport[0]), float(viewport[3] - viewport[1])
    aspect_ratio = float(W) / float(H)

    fovy = make_list2np(camera['fovy'])
    focal_length = make_list2np(camera['focal_length'])
    h = np.tan(fovy / 2) * 2 * focal_length
    w = h * aspect_ratio

    px_coord = torch.zeros_like(surfels_plane)
    px_coord[...,2] = surfels_plane[...,2] # Make sure to also transmit the new depth
    px_coord[...,:2] = surfels_plane[...,:2] * tch_var_f([-(W - 1) / w, (H - 1) / h]).unsqueeze(-2) + tch_var_f([W / 2., H / 2.]).unsqueeze(-2)
    px_coord_idx = torch.round(px_coord - 0.5).long()

    px_idx = px_coord_idx[..., 1] * W + px_coord_idx[..., 0]

    max_idx = W * H # Index used if the indices are out of bounds of the camera
    max_idx_tensor = tch_var_l([max_idx])

    # Map out of bounds pixels to the last (extra) index
    mask = (px_coord_idx[..., 1] < 0) | (px_coord_idx[..., 0] < 0) | (px_coord_idx[..., 1] >= H) | (px_coord_idx[..., 0] >= W)
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
    px_idx, _ = project_image_coordinates(surfels, camera)
    viewport = make_list2np(camera['viewport'])
    W = int(viewport[2] - viewport[0])
    rgb_reshaped = rgb.view(rgb.size(0), -1, rgb.size(-1))
    rgb_out, mask = scatter_mean_dim0(rgb_reshaped, px_idx.long())
    return rgb_out.reshape(rgb.shape), mask.reshape(rgb.shape)

def projection_renderer_differentiable(surfels, rgb, camera, rotated_image=None, blur_size=0.15):
    """Project surfels given in world coordinate to the camera's projection plane
       in a way that is differentiable w.r.t depth. This is achieved by interpolating
       the surfel values using a Gaussian filter.

    Args:
        surfels: [batch_size, num_surfels, pos]
        rgb: [batch_size, num_surfels, D-channel data] or [batch_size, H, W, D-channel data]
        camera: [{'eye': [num_batches,...], 'lookat': [num_batches,...], 'up': [num_batches,...],
                    'viewport': [0, 0, W, H], 'fovy': <radians>}]
        rotated_image: [batch_size, num_surfels, D-channel data] or [batch_size, H, W, D-channel data]
                        Image to mix in with the result of the rotation.
        sigma: Std of the Gaussian used for filtering. As a rule of thumb, surfels in a radius of 3*sigma
               around a pixel will have a contribution on that pixel in the final image.

    Returns:
        RGB image of dimensions [batch_size, H, W, 3] from projected surfels

    """
    px_idx, px_coord = project_image_coordinates(surfels, camera)
    viewport = make_list2np(camera['viewport'])
    W = int(viewport[2] - viewport[0])
    H = int(viewport[3] - viewport[1])
    rgb_reshaped = rgb.view(rgb.size(0), -1, rgb.size(-1))

    # Perform a weighted average of points surrounding a pixel using a Gaussian filter
    # Very similar to the idea in this paper: https://arxiv.org/pdf/1810.09381.pdf

    x, y = np.meshgrid(np.linspace(0, W - 1, W) + 0.5, np.linspace(0, H - 1, H) + 0.5)
    x, y = tch_var_f(x.ravel()).repeat(surfels.size(0), 1), tch_var_f(y.ravel()).repeat(surfels.size(0), 1)
    x, y = x.unsqueeze(-1), y.unsqueeze(-1)

    xp, yp = px_coord[...,0].unsqueeze(-2), px_coord[...,1].unsqueeze(-2)
    sigma = blur_size * rgb.size(-2) / 6
    scale = torch.exp((-(xp - x)**2 - (yp - y)**2) / (2 * sigma**2))

    mask = scale.sum(-1)
    if rotated_image is not None:
        rotated_image = rotated_image.view(*rgb_reshaped.size())
        # out = (rotated_image_weight * rotated_image + torch.sum(scale.unsqueeze(-1) * rgb_reshaped.unsqueeze(-3), -2)) / (scale.sum(-1) + rotated_image_weight + 1e-10).unsqueeze(-1)
        out = torch.sum(scale.unsqueeze(-1) * rgb_reshaped.unsqueeze(-3), -2) + rotated_image * (1 - mask)
    else:
        out = torch.sum(scale.unsqueeze(-1) * rgb_reshaped.unsqueeze(-3), -2) / (mask + 1e-10).unsqueeze(-1)

    return out.view(*rgb.size()), mask.view(*rgb.size()[:-1], 1)


def blur(image, blur_size, kernel=None):
    sigma = blur_size * image.size(-2) / 6
    half_kernel_size = math.floor(sigma * 3)
    if kernel is None:
        num_channels = image.size(-3)
        x_range = torch.arange(-half_kernel_size, half_kernel_size + 1, device=image.device).float()
        kernel = torch.exp(-x_range**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum() # Normalize
        kernel = torch.eye(num_channels, device=kernel.device).unsqueeze(-1).unsqueeze(-1) * kernel.view(1, 1, 1, -1)
    # Mirror padding + 2 1D convolutions of the Gaussian kernel
    padded = torch.nn.functional.pad(image, (half_kernel_size, half_kernel_size, half_kernel_size, half_kernel_size), mode='constant') # zero padding
    blurred = torch.nn.functional.conv2d(padded, kernel)
    return torch.nn.functional.conv2d(blurred, kernel.transpose(-1, -2))


def projection_renderer_differentiable_fast(surfels, rgb, camera, rotated_image=None, blur_size=0.15, use_depth=True, use_center_dist=True, compute_new_depth=False, blur_rotated_image=True):
    """Project surfels given in world coordinate to the camera's projection plane
       in a way that is differentiable w.r.t depth. This is achieved by interpolating
       the surfel values using bilinear interpolation then blurring the output image using a Gaussian filter.

    Args:
        surfels: [batch_size, num_surfels, pos] - world coordinates
        rgb: [batch_size, num_surfels, D-channel data] or [batch_size, H, W, D-channel data]
        camera: [{'eye': [num_batches,...], 'lookat': [num_batches,...], 'up': [num_batches,...],
                    'viewport': [0, 0, W, H], 'fovy': <radians>}]
        rotated_image: [batch_size, num_surfels, D-channel data] or [batch_size, H, W, D-channel data]
                        Image to mix in with the result of the rotation.
        blur_size: (between 0 and 1). Determines the size of the gaussian kernel as a percentage of the width of the input image
                   The standard deviation of the Gaussian kernel is automatically calculated from this value
        use_depth: Whether to weight the surfels landing on the same output pixel by their depth relative to the camera
        use_center_dist: Whether to weight the surfels landing on the same output pixel by their distance to the nearest pixel center location
        compute_new_depth: Whether to compute and output the depth as seen by the new camera
        blur_rotated_image: Whether to blur the 'rotated_image' passed as argument before merging it with the output image.
                            Set to False if the rotated image is already blurred

    Returns:
        RGB image of dimensions [batch_size, H, W, 3] from projected surfels
    """
    _, px_coord = project_image_coordinates(surfels, camera)
    viewport = make_list2np(camera['viewport'])
    W = int(viewport[2] - viewport[0])
    H = int(viewport[3] - viewport[1])
    rgb_in = rgb.view(rgb.size(0), -1, rgb.size(-1))

    # First create a uniform grid through bilinear interpolation
    # Then, perform a convolution with a Gaussian kernel to blur the output image
    # Idea from this paper: https://arxiv.org/pdf/1810.09381.pdf
    # Tensorflow implementation: https://github.com/eldar/differentiable-point-clouds/blob/master/dpc/util/point_cloud.py#L60

    px_idx = torch.floor(px_coord[...,:2] - 0.5).long()

    # Difference to the nearest pixel center on the top left
    x = (px_coord[...,0] - 0.5) - px_idx[...,0].float()
    y = (px_coord[...,1] - 0.5) - px_idx[...,1].float()
    x, y = x.unsqueeze(-1), y.unsqueeze(-1)

    def flat_px(px):
        """Flatten the pixel locations and make sure everything is within bounds"""
        out = px[...,1] * W + px[...,0]
        max_idx = tch_var_l([W * H])
        mask = (px[...,1] < 0) | (px[...,0] < 0) | (px[...,1] >= H) | (px[...,0] >= W)
        out = torch.where(mask, max_idx, out)
        return out

    depth = px_coord[...,2]
    center_dist_2 = (x**2 + y**2).squeeze(-1) # squared distance to the nearest pixel center
    rgb_out = scatter_weighted_blended_oit(rgb_in * (1 - x) * (1 - y), depth, center_dist_2, flat_px(px_idx + tch_var_l([0, 0])), use_depth=use_depth, use_center_dist=use_center_dist)
    rgb_out += scatter_weighted_blended_oit(rgb_in * (1 - x) * y, depth, center_dist_2, flat_px(px_idx + tch_var_l([0, 1])), use_depth=use_depth, use_center_dist=use_center_dist)
    rgb_out += scatter_weighted_blended_oit(rgb_in * x * (1 - y), depth, center_dist_2, flat_px(px_idx + tch_var_l([1, 0])), use_depth=use_depth, use_center_dist=use_center_dist)
    rgb_out += scatter_weighted_blended_oit(rgb_in * x * y, depth, center_dist_2, flat_px(px_idx + tch_var_l([1, 1])), use_depth=use_depth, use_center_dist=use_center_dist)

    soft_mask = scatter_weighted_blended_oit((1 - x) * (1 - y), depth, center_dist_2, flat_px(px_idx + tch_var_l([0, 0])), use_depth=use_depth, use_center_dist=use_center_dist)
    soft_mask += scatter_weighted_blended_oit((1 - x) * y, depth, center_dist_2, flat_px(px_idx + tch_var_l([0, 1])), use_depth=use_depth, use_center_dist=use_center_dist)
    soft_mask += scatter_weighted_blended_oit(x * (1 - y), depth, center_dist_2, flat_px(px_idx + tch_var_l([1, 0])), use_depth=use_depth, use_center_dist=use_center_dist)
    soft_mask += scatter_weighted_blended_oit(x * y, depth, center_dist_2, flat_px(px_idx + tch_var_l([1, 1])), use_depth=use_depth, use_center_dist=use_center_dist)

    if compute_new_depth:
        depth_in = depth.unsqueeze(-1)
        depth_out = scatter_weighted_blended_oit(depth_in * (1 - x) * (1 - y), depth, center_dist_2, flat_px(px_idx + tch_var_l([0, 0])), use_depth=use_depth, use_center_dist=use_center_dist)
        depth_out += scatter_weighted_blended_oit(depth_in * (1 - x) * y, depth, center_dist_2, flat_px(px_idx + tch_var_l([0, 1])), use_depth=use_depth, use_center_dist=use_center_dist)
        depth_out += scatter_weighted_blended_oit(depth_in * x * (1 - y), depth, center_dist_2, flat_px(px_idx + tch_var_l([1, 0])), use_depth=use_depth, use_center_dist=use_center_dist)
        depth_out += scatter_weighted_blended_oit(depth_in * x * y, depth, center_dist_2, flat_px(px_idx + tch_var_l([1, 1])), use_depth=use_depth, use_center_dist=use_center_dist)
        depth_out = depth_out.view(*rgb.size()[:-1], 1)

    rgb_out = rgb_out.view(*rgb.size())
    soft_mask = soft_mask.view(*rgb.size()[:-1], 1)

    # Blur the rgb and mask images
    rgb_out = blur(rgb_out.permute(0, 3, 1, 2), blur_size).permute(0, 2, 3, 1)
    soft_mask = blur(soft_mask.permute(0, 3, 1, 2), blur_size).permute(0, 2, 3, 1)

    # There seems to be a bug in PyTorch where if a single division by 0 occurs in a tensor, the whole thing becomes NaN?
    # Might be related to this issue: https://github.com/pytorch/pytorch/issues/4132
    # Because of this behavior, one can't simply do `out / out_mask` in `torch.where`
    soft_mask_nonzero = torch.where(soft_mask > 0, soft_mask, torch.ones_like(soft_mask)) + 1e-20

    # If an additional image is passed in, merge it using the soft mask:
    rgb_out_normalized = torch.where(soft_mask > 0, rgb_out / soft_mask_nonzero, rgb_out)
    if rotated_image is not None:
        if blur_rotated_image:
            rotated_image = blur(rotated_image.permute(0, 3, 1, 2), blur_size).permute(0, 2, 3, 1)
        out = torch.where(soft_mask > 1, rgb_out / soft_mask_nonzero, rgb_out + rotated_image * (1 - soft_mask))
    else:
        out = rgb_out_normalized

    # Other things to output:
    proj_out = {
        'mask': soft_mask,
        'image1': rgb_out_normalized
    }

    if compute_new_depth:
        depth_out = torch.where(soft_mask > 0, depth_out / soft_mask_nonzero, depth_out)
        proj_out['depth'] = depth_out

    return out, proj_out


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
    new_thetas = camera_eye[...,0] - theta_samples # Flip the theta samples to allow users to enter a positive value for theta range
    new_phis = camera_eye[...,1] + phi_samples

    # Go back to cartesian coordinates and place the camera back relative to the 'lookat' position
    camera_eye = spherical_to_cartesian(new_thetas, new_phis, radius=np.expand_dims(camera_eye[...,2], -1))

    if camera['at'].shape[-1] == 4:
        zeros = np.zeros((camera_eye.shape[0], 1))
        camera_eye = np.concatenate((camera_eye, zeros), axis=-1)

    camera['eye'] = tch_var_f(camera_eye) + camera['at']


def uniformly_rotate_cameras(camera, theta_range=[-np.pi / 2, np.pi / 2], phi_range=[-np.pi, np.pi]):
    """
    Given a batch of camera positions, rotate the 'eye' properties around the 'lookat' position uniformly
    for the given angle ranges (equiangular samples)
    :param camera: [{'eye': [num_batches,...], 'lookat': [num_batches,...], 'up': [num_batches,...],
                    'viewport': [0, 0, W, H], 'fovy': <radians>}]

    Modifies the camera object in place

    ASSUMES A SQUARE NUMBER OF CAMERAS if both theta_range and phi_range are not None
    """
    num_cameras = get_data(camera['eye']).shape[0]

    # Sample a theta and phi to add to the current camera rotation
    if theta_range is not None and phi_range is not None:
        width = int(np.sqrt(num_cameras))
        phi_samples, theta_samples = np.meshgrid(np.linspace(*phi_range, width), np.linspace(*theta_range, width))
        theta_samples, phi_samples = theta_samples.ravel(), phi_samples.ravel()
    elif theta_range is not None:
        theta_samples = np.linspace(*theta_range, num_cameras)
        phi_samples = np.zeros(theta_samples.shape)
    elif phi_range is not None:
        phi_samples = np.linspace(*phi_range, num_cameras)
        theta_samples = np.zeros(phi_samples.shape)

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
    return np.stack((np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)), axis=-1) * radius


def cartesian_to_spherical(coords):
    x, y, z = coords[...,0], coords[...,1], coords[...,2]
    norm2 = x**2 + z**2
    return np.stack((np.arctan2(np.sqrt(norm2), y), np.arctan2(z, x), np.sqrt(norm2 + y**2)), axis=-1)


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

    pos_wc = res['pos'].reshape(-1, res['pos'].shape[-1]).repeat(batch_size, 1, 1)

    depth = res['depth'].repeat(batch_size, 1, 1).reshape(batch_size, -1)
    # pos_cc = z_to_pcl_CC_batched(-depth, scene['camera']) # NOTE: z = -depth
    # pos_wc = cam_to_world_batched(pos_cc, None, scene['camera'])['pos']

    randomly_rotate_cameras(camera, theta_range=[-np.pi / 16, np.pi / 16], phi_range=[-np.pi / 8, np.pi / 8])

    image = res['image'].repeat(batch_size, 1, 1, 1)
    save_image(image.clone().permute(0, 3, 1, 2), 'test-original.png', nrow=2)
    save_image(depth.view(*image.size()[:-1], 1).clone().permute(0, 3, 1, 2), 'test-original-depth.png', nrow=2, normalize=True)

    # im, _ = projection_renderer(pos_wc, image, camera)
    # save_image(im.clone().permute(0, 3, 1, 2), 'test-rotated-reprojected-nonblurred.png', nrow=2)

    # If we want to merge with another already rotated image
    # NOTE: only works on batch 1 because `render` is not batched
    rotated_scene = copy.deepcopy(scene)
    rotated_scene['camera'] = copy.deepcopy(camera)
    rotated_scene['camera']['eye'] = rotated_scene['camera']['eye'][0]
    rotated_scene['camera']['at'] = rotated_scene['camera']['at'][0]
    rotated_scene['camera']['up'] = rotated_scene['camera']['up'][0]
    res_rotated = render(rotated_scene)
    rotated_image = res_rotated['image'].repeat(batch_size, 1, 1, 1)

    save_image(rotated_image.clone().permute(0, 3, 1, 2), 'test-original-rotated.png', nrow=2)

    im, proj_out = projection_renderer_differentiable_fast(pos_wc, image, camera, blur_size=1e-10, compute_new_depth=True)
    save_image(im.clone().permute(0, 3, 1, 2), 'test-fast-rotated-reprojected-unmerged.png', nrow=2)
    save_image(proj_out['mask'].clone().permute(0, 3, 1, 2), 'test-fast-soft-mask.png', nrow=2, normalize=True)
    save_image(proj_out['depth'].clone().permute(0, 3, 1, 2), 'test-fast-depth.png', nrow=2, normalize=True)

    for key in proj_out.keys():
        proj_out[key] = proj_out[key].cpu().numpy()
    np.savez('test-fast-rotation-reprojection.npz', **proj_out,
             image=im.cpu().numpy(), image_in=image.cpu().numpy(), depth_in=depth.view(*image.size()[:-1], 1).cpu().numpy())

    im, _ = projection_renderer_differentiable_fast(pos_wc, image, camera, rotated_image, blur_size=1e-10)
    save_image(im.clone().permute(0, 3, 1, 2), 'test-fast-rotated-reprojected-merged.png', nrow=2)


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
                      out_dir='./proj_tmp_depth-fast/'):
    """ First render using the full renderer to get the surfel position and color
    and then render using the projection layer for testing

    Returns:

    """
    from torch import optim
    import torchvision
    import os
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import imageio
    from PIL import Image
    plt.ion()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    use_chair = True
    use_fast_projection = True
    use_masked_loss = True
    use_same_render_method_for_target = False
    lr = 1e-2

    res = render_scene(scene)

    scene = make_torch_var(load_scene(scene))
    # true_pos_wc = res['pos'].reshape(-1, res['pos'].shape[-1]).repeat(batch_size, 1, 1)
    true_input_img = res['image'].unsqueeze(0).repeat(batch_size, 1, 1, 1)

    if use_chair:
        camera = scene['camera']
        camera['eye'] = tch_var_f([0, 0, 4, 1]).repeat(batch_size, 1)
        camera['at'] = tch_var_f([0, 0, 0, 1]).repeat(batch_size, 1)
        camera['up'] = tch_var_f([0, 1, 0, 0]).repeat(batch_size, 1)
    else:
        camera = scene['camera']
        camera['eye'] = camera['eye'].repeat(batch_size, 1)
        camera['at'] = camera['at'].repeat(batch_size, 1)
        camera['up'] = camera['up'].repeat(batch_size, 1)

    if use_chair:
        chair_0 = Image.open('object-0-azimuth-000.006-rgb.png')
        true_input_img = torchvision.transforms.ToTensor()(chair_0).to(true_input_img.device).unsqueeze(0)
        true_input_img = true_input_img.permute(0, 2, 3, 1)
        camera['viewport'] = [0, 0, 128, 128] # TODO don't hardcode

    true_depth = res['depth'].repeat(batch_size, 1, 1).reshape(batch_size, -1) # Not relevant if 'use_chair' is True
    # depth = true_depth.clone() + 0.1 * torch.randn_like(true_depth)
    depth = 0.1 * torch.randn(batch_size, true_input_img.size(-2) * true_input_img.size(-3), device=true_input_img.device, dtype=torch.float)
    depth.requires_grad = True

    if use_chair:
        target_angle = np.deg2rad(20)
    else:
        target_angle = -np.pi / 12

    rotated_camera = copy.deepcopy(camera)
    # randomly_rotate_cameras(rotated_camera, theta_range=[-np.pi / 16, np.pi / 16], phi_range=[-np.pi / 8, np.pi / 8])
    randomly_rotate_cameras(rotated_camera, theta_range=[0, 1e-10], phi_range=[target_angle, target_angle + 1e-10])

    if use_chair:
        target_image = Image.open('object-0-azimuth-020.006-rgb.png')
        target_image = torchvision.transforms.ToTensor()(target_image).to(true_input_img.device).unsqueeze(0)
        target_image = target_image.permute(0, 2, 3, 1)
        target_mask = torch.ones(*target_image.size()[:-1], 1, device=target_image.device, dtype=torch.float)
    else:
        true_pos_cc = z_to_pcl_CC_batched(-true_depth, camera) # NOTE: z = -depth
        true_pos_wc = cam_to_world_batched(true_pos_cc, None, camera)['pos']

        if use_same_render_method_for_target:
            if use_fast_projection:
                target_image, proj_out = projection_renderer_differentiable_fast(true_pos_wc, true_input_img, rotated_camera)
                target_mask = proj_out['mask']
            else:
                target_image, target_mask = projection_renderer_differentiable(true_pos_wc, true_input_img, rotated_camera)
            # target_image, _ = projection_renderer(true_pos_wc, true_input_img, rotated_camera)
        else:
            scene2 = copy.deepcopy(scene)
            scene['camera'] = copy.deepcopy(rotated_camera)
            scene['camera']['eye'] = scene['camera']['eye'][0]
            scene['camera']['at'] = scene['camera']['at'][0]
            scene['camera']['up'] = scene['camera']['up'][0]
            target_image = render(scene)['image'].unsqueeze(0).repeat(batch_size, 1, 1, 1)
            target_mask = torch.ones(*target_image.size()[:-1], 1, device=target_image.device, dtype=torch.float)

    input_image = true_input_img # + 0.1 * torch.randn(target_image.size(), device=target_image.device)

    criterion = torch.nn.MSELoss(reduction='none').cuda()
    optimizer = optim.Adam([depth], lr=1e-2)

    h1 = plt.figure()
    # fig_imgs = []
    depth_imgs = []
    out_imgs = []

    imageio.imsave(out_dir + '/optimization_input_image.png', input_image[0].cpu().numpy())
    imageio.imsave(out_dir + '/optimization_target_image.png', target_image[0].cpu().numpy())
    if not use_chair:
        imageio.imsave(out_dir + '/optimization_target_depth.png', true_depth.view(*input_image.size()[:-1], 1)[0].cpu().numpy())

    loss_per_iter = []
    for iter in range(500):
        optimizer.zero_grad()
        # depth_in = torch.nn.functional.softplus(depth + 3)
        depth_in = depth + 4
        pos_cc = z_to_pcl_CC_batched(-depth_in, camera) # NOTE: z = -depth
        pos_wc = cam_to_world_batched(pos_cc, None, camera)['pos']
        if use_fast_projection:
            im_est, proj_out = projection_renderer_differentiable_fast(pos_wc, input_image, rotated_camera)
            im_mask = proj_out['mask']
        else:
            im_est, im_mask = projection_renderer_differentiable(pos_wc, input_image, rotated_camera)
        # im_est, mask = projection_renderer(pos_wc, input_image, rotated_camera)
        if use_masked_loss:
            loss = torch.sum(target_mask * im_mask * criterion(im_est * 255, target_image * 255)) / torch.sum(target_mask * im_mask)
        else:
            loss = criterion(im_est * 255, target_image * 255).mean()
        loss_ = get_data(loss)
        loss_per_iter.append(loss_)
        if iter % print_interval == 0 or iter == max_iter - 1:
            print('{}. Loss: {}'.format(iter, loss_))
        if iter % imsave_interval == 0 or iter == max_iter - 1:
            # Input image
            # im_out_ = get_data(input_image.detach())
            # im_out_ = np.uint8(255 * im_out_ / im_out_.max())
            # fig = plt.figure(h1.number)
            # plot = fig.add_subplot(111)
            # plot.imshow(im_out_[0].squeeze())
            # plot.set_title('%d. loss= %f' % (iter, loss_))
            # # plt.savefig(out_dir + '/fig_%05d.png' % iter)
            # fig_data = np.array(fig.canvas.renderer._renderer)
            # fig_imgs.append(fig_data)

            # Depth
            im_out_ = get_data(depth_in.view(*input_image.size()[:-1], 1).detach())
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

        loss.backward()
        optimizer.step()
    
    # imageio.mimsave(out_dir + '/optimization_anim_in.gif', fig_imgs)
    imageio.mimsave(out_dir + '/optimization_anim_depth.gif', depth_imgs)
    imageio.mimsave(out_dir + '/optimization_anim_out.gif', out_imgs)


def test_depth_to_world_consistency(scene, batch_size):
    res = render_scene(scene)

    scene = make_torch_var(load_scene(scene))

    pos_wc1 = res['pos'].reshape(-1, res['pos'].shape[-1])

    pos_cc1 = world_to_cam(pos_wc1, None, scene['camera'])['pos']
    # First test the non-batched z_to_pcl_CC method:
    # NOTE: z_to_pcl_CC takes as input the Z dimension in the camera coordinate
    # and gets the full (X, Y, Z) in the camera coordinate.
    pos_cc2 = z_to_pcl_CC(pos_cc1[:, 2], scene['camera'])
    pos_wc2 = cam_to_world(pos_cc2, None, scene['camera'])['pos']

    # Test Z -> (X, Y, Z)
    np.testing.assert_array_almost_equal(get_data(pos_cc1[..., :3]), get_data(pos_cc2[..., :3]))
    # Test world -> camera -> Z -> (X, Y, Z) in camera -> world
    np.testing.assert_array_almost_equal(get_data(pos_wc1[..., :3]), get_data(pos_wc2[..., :3]))

    # Then test the batched version:
    camera = scene['camera']
    camera['eye'] = camera['eye'].repeat(batch_size, 1)
    camera['at'] = camera['at'].repeat(batch_size, 1)
    camera['up'] = camera['up'].repeat(batch_size, 1)

    pos_wc1 = pos_wc1.repeat(batch_size, 1, 1)
    pos_cc1 = world_to_cam_batched(pos_wc1, None, scene['camera'])['pos']
    pos_cc2 = z_to_pcl_CC_batched(pos_cc1[..., 2], camera) # NOTE: z = -depth
    pos_wc2 = cam_to_world_batched(pos_cc2, None, camera)['pos']

    # Test Z -> (X, Y, Z)
    np.testing.assert_array_almost_equal(get_data(pos_cc1[..., :3]), get_data(pos_cc2[..., :3]))
    # Test world -> camera -> Z -> (X, Y, Z) in camera -> world
    np.testing.assert_array_almost_equal(get_data(pos_wc1[..., :3]), get_data(pos_wc2[..., :3]))


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

    # test_depth_to_world_consistency(scene, batch_size)
    # test_visual_render(scene, 1)
    # test_render_projection_consistency(scene, batch_size)
    # test_raster_coordinates(scene, batch_size)
    # test_transformation_consistency(scene, batch_size)
    # test_optimization(scene, 1)
    test_depth_optimization(scene, 1)


if __name__ == '__main__':
    main()
