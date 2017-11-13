import numpy as np
import torch

from diffrend.torch.utils import tonemap, ray_object_intersections, generate_rays


class Renderer():
    def __init__(self):
        super(Renderer, self).__init__()

    def render(self, scene):
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
        pixel_dist[~valid_pixels] = np.inf  # Will have to use gather operation for TF and pytorch

        # Nearest object needs to be compared for valid regions only
        nearest_obj = np.argmin(pixel_dist, axis=0)
        C = np.arange(0, nearest_obj.size)  # pixel idx

        # Create depth image for visualization
        # use nearest_obj for gather/select the pixel color
        im_depth = pixel_dist[nearest_obj, C].reshape(H, W)

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
        frag_normals = normals[nearest_obj, C]
        frag_pos = obj_intersections[nearest_obj, C]
        frag_albedo = scene['materials']['albedo'][material_idx[nearest_obj]]

        # Fragment shading
        light_dir = light_pos[np.newaxis, :] - frag_pos[:, np.newaxis, :]
        light_dir_norm = np.sqrt(np.sum(light_dir ** 2, axis=-1))[..., np.newaxis]
        # light_dir_norm[light_dir_norm <= 0 | np.isinf(light_dir_norm)] = 1
        light_dir /= light_dir_norm
        im_color = np.sum(frag_normals[:, np.newaxis, :] * light_dir, axis=-1)[..., np.newaxis] * \
                   light_colors[np.newaxis, ...] * frag_albedo[:, np.newaxis, :]

        im = np.sum(im_color, axis=1).reshape(H, W, 3)
        im[(im_depth < camera['near']) | (im_depth > camera['far'])] = 0

        # clip negative values
        im[im < 0] = 0

        # Tonemapping
        if 'tonemap' in scene:
            im = tonemap(im, **scene['tonemap'])

        return {
            'image': im,
            'depth': im_depth,
            'ray_dist': ray_dist,
            'obj_dist': pixel_dist,
            'nearest': nearest_obj.reshape(H, W),
            'ray_dir': ray_dir,
            'valid_pixels': valid_pixels
        }
