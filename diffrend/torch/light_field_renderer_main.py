from diffrend.torch.utils import tch_var_f
from diffrend.torch.LightFieldNet import LFNetV0
from diffrend.torch.render import render_scene
from diffrend.utils.sample_generator import uniform_sample_sphere
import numpy as np
import torch
import matplotlib.pyplot as plt


def lf_renderer(pos, normal, lfnet, num_samples=10):
    pos_all = pos.reshape((-1, 3))
    normal_all = normal.reshape((-1, 3))
    pixel_colors = []
    for idx in range(pos_all.shape[0]):
        dir_sample = uniform_sample_sphere(radius=1.0, num_samples=num_samples, axis=normal_all[idx], angle=np.pi / 2)
        inp = tch_var_f(np.concatenate((np.tile(pos_all[idx], (num_samples, 1)),
                                        np.tile(normal_all[idx], (num_samples, 1)),
                                        dir_sample), axis=-1))
        Li = lfnet(inp)
        cos_theta = torch.sum(inp[:, 3:6] * inp[:, 6:], dim=-1)
        rgb = torch.sum(cos_theta[:, np.newaxis] * Li, dim=0)
        pixel_colors.append(rgb)

    im = torch.cat(pixel_colors, dim=0).reshape(pos.shape)
    return im


scene_file = '../../scenes/halfbox_sphere_cube.json'
res = render_scene(scene_file)

img = res['image'].cpu().numpy().squeeze()
pos = res['pos'].cpu().numpy().squeeze()
normal = res['normal'].cpu().numpy().squeeze()
depth = res['depth'].cpu().numpy().squeeze()

lfnet = LFNetV0(in_ch=9, out_ch=3).cuda()
num_samples = 10
im = lf_renderer(pos, normal, lfnet, num_samples=num_samples)
im_ = im.cpu().detach().numpy()
plt.figure()
plt.imshow(im_)

# TODO: Optimize lfnet
# ..

plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(normal)