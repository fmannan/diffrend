from __future__ import division
import numpy as np
import torch
from diffrend.torch.utils import (tonemap, ray_object_intersections,
                                  generate_rays, where, backface_labeler,
                                  bincount, tch_var_f, norm_p, normalize,
                                  lookat)
from diffrend.utils.utils import get_param_value
from diffrend.torch.ops import perspective, inv_perspective
"""
Scalable Rendering TODO:
1. Backface culling. Cull splats for which dot((eye - pos), normal) <= 0 [DONE]
1.1. Filter out objects based on backface labeling.
2. Frustum culling
3. Ray culling: Low-res image and per-pixel frustum culling to determine the
   valid rays
4. Bound sphere for splats
5. OpenGL pass to determine visible splats. I.e. every pixel in the output
   image will have the splat index, the intersection point
6. Specialized version that does not render any non-planar geometry. For these the 
normals per pixel do not need to be stored.
Implement ray_planar_object_intersection
"""


def render(scene, **params):
    """Render.

    :param scene: Scene description
    :return: [H, W, 3] image
    """
    # Construct rays from the camera's eye position through the screen
    # coordinates
    camera = scene['camera']
    ray_orig, ray_dir, H, W = generate_rays(camera)
    H = int(H)
    W = int(W)

    scene_objects = scene['objects']

    if get_param_value('backface_culling', params, False):
        # Add a binary label per planar geometry.
        # 1: Facing away from the camera, i.e., back-face, i.e., dot(camera_dir, normal) < 0
        # 0: Facing the camera.
        # Labels are stored in the key 'backface'
        # Note that doing this before ray object intersection test reduces memory but may not result in correct
        # rendering, e.g, when an object is occluded by a back-face.
        scene_objects = backface_labeler(ray_orig, scene_objects)

    # Ray-object intersections
    disable_normals = get_param_value('norm_depth_image_only', params, False)
    obj_intersections, ray_dist, normals, material_idx = ray_object_intersections(ray_orig, ray_dir, scene_objects,
                                                                                  disable_normals=disable_normals)
    num_objects = obj_intersections.size()[0]
    # Valid distances
    pixel_dist = ray_dist
    valid_pixels = (camera['near'] <= ray_dist) * (ray_dist <= camera['far'])
    pixel_dist = where(valid_pixels, pixel_dist, camera['far'] + 1)

    # Nearest object depth and index
    im_depth, nearest_obj = pixel_dist.min(0)

    # Reshape to image for visualization
    # use nearest_obj for gather/select the pixel color
    # im_depth = torch.gather(pixel_dist, 0, nearest_obj[np.newaxis, :]).view(H, W)
    im_depth = im_depth.view(H, W)

    # Find the number of pixels covered by each object
    pixel_obj_count = torch.sum(valid_pixels, dim=0)
    valid_pixels_mask = pixel_obj_count > 0
    nearest_obj_only = torch.masked_select(nearest_obj, valid_pixels_mask)
    obj_pixel_count = bincount(nearest_obj_only, num_objects)

    if get_param_value('norm_depth_image_only', params, False):
        min_depth = torch.min(im_depth)
        norm_depth_image = where(im_depth >= camera['far'], min_depth, im_depth)
        norm_depth_image = (norm_depth_image - min_depth) / (torch.max(im_depth) - min_depth)
        return {
            'image': norm_depth_image,
            'depth': im_depth,
            'ray_dist': ray_dist,
            'obj_dist': pixel_dist,
            'nearest': nearest_obj.view(H, W),
            'ray_dir': ray_dir,
            'valid_pixels': valid_pixels,
            'obj_pixel_count': obj_pixel_count,
            'pixel_obj_count': pixel_obj_count,
            'valid_pixels_mask': valid_pixels_mask.view(H, W),
        }

    ##############################
    # Fragment processing
    ##############################
    # Lighting
    color_table = scene['colors']
    light_pos = scene['lights']['pos'][:, :3]
    light_clr_idx = scene['lights']['color_idx']
    light_colors = color_table[light_clr_idx]

    materials = scene['materials']['albedo']

    # Generate the fragments
    """
    Get the normal and material for the visible objects.
    """
    frag_normals = torch.gather(
        normals, 0, nearest_obj[np.newaxis, :, np.newaxis].repeat(1, 1, 3))
    frag_pos = torch.gather(
        obj_intersections, 0,
        nearest_obj[np.newaxis, :, np.newaxis].repeat(1, 1, 3))
    tmp_idx = torch.gather(material_idx, 0, nearest_obj)
    frag_albedo = torch.index_select(materials, 0, tmp_idx)

    # Fragment shading
    light_dir = light_pos[:, np.newaxis, :] - frag_pos
    light_dir_norm = torch.sqrt(torch.sum(light_dir ** 2, dim=-1))[:, :, np.newaxis]
    light_dir /= light_dir_norm  # TODO: nonzero_divide

    im_color = torch.sum(frag_normals * light_dir, dim=-1)[:, :, np.newaxis] * \
               light_colors[:, np.newaxis, :] * frag_albedo[np.newaxis, :, :]

    im = torch.sum(im_color, dim=0).view(int(H), int(W), 3)

    valid_pixels = (camera['near'] <= im_depth) * (im_depth <= camera['far'])
    im = valid_pixels[:, :, np.newaxis].float() * im

    # clip non-negative
    im = torch.nn.functional.relu(im)

    # Tonemapping
    if 'tonemap' in scene:
        im = tonemap(im, **scene['tonemap'])

    return {
        'image': im,
        'depth': im_depth,
        'ray_dist': ray_dist,
        'obj_dist': pixel_dist,
        'nearest': nearest_obj.view(H, W),
        'ray_dir': ray_dir,
        'valid_pixels': valid_pixels,
        'obj_pixel_count': obj_pixel_count,
        'pixel_obj_count': pixel_obj_count,
        'valid_pixels_mask': valid_pixels_mask.view(H, W),
    }


def render_splats_NDC(scene, **params):
    """Render splats specified in the camera's normalized coordinate system

    For now, assume number of splats to be the number of pixels This would be relaxed later to allow subpixel rendering.
    :param scene: Scene description
    :return: [H, W, 3] image
    """
    camera = scene['camera']
    viewport = np.array(camera['viewport'])
    W, H = int(viewport[2] - viewport[0]), int(viewport[3] - viewport[1])
    aspect_ratio = W / H
    fovy = camera['fovy']
    near = camera['near']
    far = camera['far']
    eye = camera['eye'][:3]
    at = camera['at'][:3]
    up = camera['up'][:3]
    Mcam = lookat(eye=eye, at=at, up=up)
    M = perspective(fovy, aspect_ratio, near, far)
    Minv = inv_perspective(fovy, aspect_ratio, near, far)

    splats = scene['objects']['disk']
    pos_NDC = splats['pos']
    normals_SLC = splats['normal']
    num_objects = pos_NDC.size()[0]

    # Transform params to the Camera's view frustum
    if pos_NDC.size()[-1] == 3:
        pos_NDC = torch.cat((pos_NDC, tch_var_f(np.ones((num_objects, 1)))), dim=1)
    pos_CC = torch.matmul(pos_NDC, Minv.transpose(1, 0))
    pos_CC = pos_CC / pos_CC[..., 3][:, np.newaxis]

    im_depth = norm_p(pos_CC[..., :3]).view(H, W)

    if get_param_value('norm_depth_image_only', params, False):
        min_depth = torch.min(im_depth)
        norm_depth_image = where(im_depth >= camera['far'], min_depth, im_depth)
        norm_depth_image = (norm_depth_image - min_depth) / (torch.max(im_depth) - min_depth)
        return {
            'image': norm_depth_image,
            'depth': im_depth,
        }
    ##############################
    # Fragment processing
    # -------------------
    # We can either perform the operations in the world coordinate or in the camera coordinate
    # Since the inputs are in NDC and converted to CC, converting to world coordinate would require more operations.
    # There are fewer lights than splats, so converting light positions and directions to CC is more efficient.
    ##############################
    # Lighting
    color_table = scene['colors']
    light_pos = scene['lights']['pos']
    light_clr_idx = scene['lights']['color_idx']
    light_colors = color_table[light_clr_idx]

    materials = scene['materials']['albedo']
    material_idx = scene['objects']['disk']['material_idx']

    light_pos_CC = torch.mm(light_pos, Mcam.transpose(1, 0))

    # Generate the fragments
    """
    Get the normal and material for the visible objects.
    """
    normals_CC = normals_SLC   # TODO: NEED TO TRANSFORM TO CC
    frag_normals = normals_CC
    frag_pos = pos_CC

    frag_albedo = torch.index_select(materials, 0, material_idx)

    # Fragment shading
    light_dir = light_pos_CC[:, np.newaxis, :3] - frag_pos[:, :3]
    light_dir_norm = torch.sqrt(torch.sum(light_dir ** 2, dim=-1))[:, :, np.newaxis]
    light_dir /= light_dir_norm  # TODO: nonzero_divide

    im_color = torch.sum(frag_normals * light_dir, dim=-1)[:, :, np.newaxis] * \
               light_colors[:, np.newaxis, :] * frag_albedo[np.newaxis, :, :]

    im = torch.sum(im_color, dim=0).view(int(H), int(W), 3)

    # clip non-negative
    im = torch.nn.functional.relu(im)

    # Tonemapping
    if 'tonemap' in scene:
        im = tonemap(im, **scene['tonemap'])

    return {
        'image': im,
        'depth': im_depth,
    }


def test_render_splat_NDC_0():
    fovy = np.deg2rad(45)
    aspect_ratio = 1
    near = 0.1
    far = 1000
    M = perspective(fovy, aspect_ratio, near, far)
    Minv = inv_perspective(fovy, aspect_ratio, near, far)

    pos_NDC = tch_var_f([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    normals_SLC = tch_var_f([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    num_objects = pos_NDC.size()[0]

    # Transform params to the Camera's view frustum
    if pos_NDC.size()[-1] == 3:
        pos_NDC = torch.cat((pos_NDC, tch_var_f(np.ones((num_objects, 1)))), dim=1)
    pos_CC = torch.matmul(pos_NDC, Minv.transpose(1, 0))
    pos_CC = pos_CC / pos_CC[..., 3][:, np.newaxis]

    pixel_dist = norm_p(pos_CC[..., :3])