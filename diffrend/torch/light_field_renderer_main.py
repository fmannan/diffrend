from diffrend.torch.utils import tch_var_f, get_data
from diffrend.torch.LightFieldNet import LFNetV0
from diffrend.torch.render import render_scene
from diffrend.utils.sample_generator import uniform_sample_sphere
import numpy as np
import torch
import matplotlib.pyplot as plt


def lf_renderer_v0(pos, normal, lfnet, num_samples=10):
    pos_all = pos.reshape((-1, 3))
    normal_all = normal.reshape((-1, 3))
    pixel_colors = []

    for idx in range(pos_all.shape[0]):
        dir_sample = uniform_sample_sphere(radius=1.0, num_samples=num_samples, axis=normal_all[idx], angle=np.pi / 2)
        inp = tch_var_f(np.concatenate((np.tile(pos_all[idx], (num_samples, 1)),
                                        dir_sample), axis=-1))
        Li = lfnet(inp)
        cos_theta = torch.sum(inp[:, 3:6] * tch_var_f(normal_all[idx]), dim=-1)
        rgb = torch.sum(cos_theta[:, np.newaxis] * Li, dim=0)
        pixel_colors.append(rgb)

    im = torch.cat(pixel_colors, dim=0).reshape(pos.shape)
    return im


def lf_renderer(pos, normal, lfnet, num_samples=20):
    """This is a simpler version of lf_renderer_v0 where the same direction samples are used
    for all surfels. The samples are on a uniform sphere and so this renderer also supports
    transmissive medium.

    Args:
        pos:
        normal:
        lfnet:
        num_samples:

    Returns:

    """
    pos_all = pos.reshape((-1, 3))
    normal_all = tch_var_f(normal.reshape((-1, 3)))

    spherical_samples = uniform_sample_sphere(radius=1.0, num_samples=num_samples)

    inp = tch_var_f(np.concatenate((np.tile(pos_all[:, np.newaxis, :], (1, num_samples, 1)),
                                    np.tile(spherical_samples[np.newaxis, :, :], (pos_all.shape[0], 1, 1))),
                                   axis=-1))
    Li = lfnet(inp)
    cos_theta = torch.sum(inp[:, :, 3:6] * normal_all[:, np.newaxis, :], dim=-1)
    nonzero_mask = (cos_theta > 0).float()
    pos_cos_theta = cos_theta * nonzero_mask
    im = torch.sum(pos_cos_theta[..., np.newaxis] * Li, dim=1).reshape(pos.shape)

    return im


def optimize_lfnet(scene, lfnet, samples=20):
    pass


scene_file = '../../scenes/halfbox_sphere_cube.json'
res = render_scene(scene_file)

img = get_data(res['image'])
pos = get_data(res['pos'])
normal = get_data(res['normal'])
depth = get_data(res['depth'])

lfnet = LFNetV0(in_ch=6, out_ch=3).cuda()
num_samples = 20
im = lf_renderer(pos, normal, lfnet, num_samples=num_samples)
im_ = get_data(im)
plt.ion()
plt.figure()
plt.imshow(im_)

plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(normal)

plt.show()

# max_iter = 5
# # Optimize lfnet
# for iter in range(max_iter):
#     im = lf_renderer(pos, normal, lfnet, num_samples=num_samples)
#     im_ = get_data(im)
#     plt.figure()
#     plt.imshow(im_)



