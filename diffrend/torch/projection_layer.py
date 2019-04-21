import torch
import numpy as np
from diffrend.torch.utils import cam_to_world, world_to_cam, get_data
from diffrend.torch.render import render_scene, load_scene, make_torch_var
from diffrend.torch.utils import make_list2np, tch_var_f


"""Projection Layer
1. Project to the image plane
2. Transform to viewport
3. Find pixel index
4. index_select (gather) surfel data to pixel (TODO: currently only selects but take the per-pixel average)
"""


def project_surfels(surfel_pos_WC, camera):
    """Returns the surfel coordinates (defined in world coordinate) on a projection plane
    (in the camera's coordinate frame)

    Args:
        surfels: [batch_size, num_surfels, 3D pos]
        camera: [{'eye': [], 'lookat': [], 'up': [], 'viewport': [0, 0, W, H], 'fovy': <radians>}
                  ...num_batches...
                ]

    Returns:
        [batch_size, num_surfels, 2D pos]

    """
    # TODO (fmannan): Assuming batch_size = 1 for now. For batch_size > 1 need to do batch_transformation
    surfels_cc = world_to_cam(surfel_pos_WC, None, camera)['pos']

    """ If a 3D point (X, Y, Z) is projected to a 2D point (x, y) then for focal_length f,
    x/f = X/Z and y/f = Y/Z
    """
    focal_length = camera['focal_length']
    x = focal_length * surfels_cc[:, 0] / surfels_cc[:, 2]
    y = focal_length * surfels_cc[:, 1] / surfels_cc[:, 2]

    return torch.stack((x, y), dim=-1)


def project_image_coordinates(surfels, camera):
    """Project surfels given in world coordinate to the camera's projection plane.

    Args:
        surfels: [batch_size, pos]
                camera: [{'eye': [], 'lookat': [], 'up': [], 'viewport': [0, 0, W, H], 'fovy': <radians>}
                  ...num_batches...
                ]

    Returns:
        RGB image of dimensions [batch_size, H, W, 3] from projected surfels

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

    px_coord = surfels_2d * tch_var_f([-(W - 1) / w, (H - 1) / h])[np.newaxis, :] + tch_var_f([W / 2., H / 2.])[
                                                                                    np.newaxis, :]
    px_coord_idx = torch.round(px_coord - 0.5)

    return px_coord_idx, {'px_coord': px_coord, 'px_coord_idx': px_coord_idx}


def project_input(surfels, input, camera):
    """Project surfels given in world coordinate to the camera's projection plane.

    Args:
        surfels: [batch_size, num_surfels, pos]
        input: [batch_size, num_surfels, D-dim data]
        camera: [{'eye': [], 'lookat': [], 'up': [], 'viewport': [0, 0, W, H], 'fovy': <radians>}
                  ...num_batches...
                ]

    Returns:
        output [batch_size, num_surfels, D-dim data] from projected surfels

    """
    px_coord_idx, _ = project_image_coordinates(surfels, camera)
    viewport = make_list2np(camera['viewport'])
    W = int(viewport[2] - viewport[0])
    idx = px_coord_idx[:, 1] * W + px_coord_idx[:, 0]
    return torch.index_select(input.reshape(-1, input.shape[-1]), 0, idx.long())


def projection_renderer(surfels, rgb, camera):
    """Project surfels given in world coordinate to the camera's projection plane.

    Args:
        surfels: [batch_size, num_surfels, pos]
        rgb: [batch_size, num_surfels, D-channel data]
        camera: [{'eye': [], 'lookat': [], 'up': [], 'viewport': [0, 0, W, H], 'fovy': <radians>}
                  ...num_batches...
                ]

    Returns:
        RGB image of dimensions [batch_size, H, W, 3] from projected surfels

    """
    rgb_out = project_input(surfels, rgb.reshape(-1, 3), camera)
    return rgb_out.reshape(rgb.shape)


def test_raster_coordinates(scene):
    """Test if the projected raster coordinates are correct

    Args:
        scene: Path to scene file

    Returns:
        None

    """
    res = render_scene(scene)
    scene = make_torch_var(load_scene(scene))
    pos_cc = res['pos'].reshape(-1, res['pos'].shape[-1])
    camera = scene['camera']
    viewport = make_list2np(camera['viewport'])
    W, H = float(viewport[2] - viewport[0]), float(viewport[3] - viewport[1])
    px_coord_idx, _ = project_image_coordinates(pos_cc, scene['camera'])
    xp, yp = np.meshgrid(np.linspace(0, W - 1, int(W)), np.linspace(0, H - 1, int(H)))

    np.testing.assert_array_almost_equal(xp.ravel(), get_data(px_coord_idx[:, 0]))
    np.testing.assert_array_almost_equal(yp.ravel(), get_data(px_coord_idx[:, 1]))


def test_render_projection_consistency(scene):
    """ First render using the full renderer to get the surfel position and color
    and then render using the projection layer for testing

    Returns:

    """
    res = render_scene(scene)

    scene = make_torch_var(load_scene(scene))
    pos_cc = res['pos'].reshape(-1, res['pos'].shape[-1])
    im = projection_renderer(pos_cc, res['image'], scene['camera'])
    diff = np.abs(get_data(res['image']) - get_data(im))
    np.testing.assert_(diff.sum() < 1e-10, 'Non-zero difference.')


def test_transformation_consistency(scene):
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

    test_render_projection_consistency(scene)
    test_raster_coordinates(scene)
    test_render_projection_consistency(scene)


if __name__ == '__main__':
     main()
