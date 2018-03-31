import numpy as np
from diffrend.torch.renderer import render_splats_along_ray
from diffrend.torch.utils import get_data, tch_var_f
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


scene = np.load('scene_output_twogans.npy')
#scene = np.load('scene_output.npy')
#pos = get_data(scene[0]['objects']['disk']['pos'])

for idx in range(0, len(scene), 20):
    print(idx)
    scene[idx]['lights']['attenuation'] = tch_var_f([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    res = render_splats_along_ray(scene[idx])

    im = get_data(res['image'])
    depth = get_data(res['depth'])

    plt.figure()
    plt.imshow(im)

    plt.figure()
    plt.imshow(depth)

    pos = get_data(res['pos'])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=1.3)


# data = np.load('res_world_twogans.npy')
#
# # pos0 = get_data(data[0]['pos'])
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(pos0[:, 0], pos0[:, 1], pos0[:, 2], s=1.3)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for idx in range(0, len(data), 20):
#     if type(data[idx]['pos']) is np.ndarray:
#         pos = data[idx]['pos']
#     else:
#         pos = get_data(data[idx]['pos'])
#     ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=1.3)
#
#
# filename_prefix = 'test'
#
# for idx in range(0, len(data), 20):
#     if type(data[idx]['pos']) is np.ndarray:
#         pos = data[idx]['pos']
#     else:
#         pos = get_data(data[idx]['pos'])
#     pos = pos[:, :3]
#     with open(filename_prefix + '_{:05d}.xyz'.format(idx), 'w') as fid:
#         for sub_idx in range(pos.shape[0]):
#             fid.write('{}\n'.format(' '.join([str(x) for x in pos[sub_idx]])))


