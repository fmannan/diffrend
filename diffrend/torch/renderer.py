import numpy as np
import torch
from diffrend.torch.utils import tonemap, ray_object_intersections, generate_rays, where, tch_var_f

"""
Scalable Rendering TODO:
1. Backface culling. Cull splats for which dot((eye - pos), normal) <= 0 
2. Frustum culling
3. Ray culling: Low-res image and per-pixel frustum culling to determine the valid rays
4. Bound sphere for splats 
5. OpenGL pass to determine visible splats. I.e. every pixel in the output image will have the splat index, the 
intersection point  
"""

def render(scene):
    """
    :param scene: Scene description
    :return: [H, W, 3] image
    """
    # Construct rays from the camera's eye position through the screen coordinates
    camera = scene['camera']
    eye, ray_dir, H, W = generate_rays(camera)
    H = int(H)
    W = int(W)

    # Ray-object intersections
    scene_objects = scene['objects']
    obj_intersections, ray_dist, normals, material_idx = ray_object_intersections(eye, ray_dir, scene_objects)

    # Valid distances
    pixel_dist = ray_dist
    valid_pixels = (camera['near'] <= ray_dist) * (ray_dist <= camera['far'])
    pixel_dist = pixel_dist * valid_pixels.float() + (camera['far'] + 1) * (1 - valid_pixels.float())

    # Nearest object needs to be compared for valid regions only
    nearest_dist, nearest_obj = pixel_dist.min(0)

    # Create depth image for visualization
    # use nearest_obj for gather/select the pixel color
    im_depth = torch.gather(pixel_dist, 0, nearest_obj[np.newaxis, :]).view(H, W)

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
    frag_normals = torch.gather(normals, 0, nearest_obj[np.newaxis, :, np.newaxis].repeat(1, 1, 3))
    frag_pos = torch.gather(obj_intersections, 0, nearest_obj[np.newaxis, :, np.newaxis].repeat(1, 1, 3))
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
    }
