from diffrend.torch.params import OUTPUT_FOLDER, SCENE_BASIC
from diffrend.torch.renderer import render
from diffrend.torch.utils import tch_var_f, CUDA
import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import os


def render_scene(scene, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # main render run
    res = render(scene)
    if CUDA:
        im = res['image'].cpu().data.numpy()
    else:
        im = res['image'].data.numpy()

    plt.ion()
    plt.figure()
    plt.imshow(im)
    plt.title('Final Rendered Image')
    plt.savefig(output_folder + 'img_torch.png')

    if CUDA:
        depth = res['depth'].cpu().data.numpy()
    else:
        depth = res['depth'].data.numpy()
    depth[depth >= scene['camera']['far']] = np.inf
    plt.figure()
    plt.imshow(depth)
    plt.title('Depth Image')
    plt.savefig(output_folder + 'img_depth_torch.png')

    plt.ioff()
    plt.show()
    return res


def optimize_scene(input_scene, target_scene, out_dir, max_iter=100, lr=1e-3, print_interval=10,
                   imsave_interval=10):
    """A demo function to check if the differentiable renderer can optimize.
    :param scene:
    :param out_dir:
    :return:
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    target_res = render(target_scene)
    target_im = target_res['image']
    target_im.require_grad = False
    if CUDA:
        target_im_ = target_res['image'].cpu()

    plt.ion()
    plt.figure()
    plt.imshow(target_im_.data.numpy())
    plt.title('Target Image')
    plt.savefig(out_dir + 'target.png')

    optimizer = optim.Adam(input_scene['materials'].values(), lr=lr)
    optimizer.zero_grad()

    #h1 = plt.figure()

    for iter in range(max_iter):
        res = render(input_scene)
        im_out = res['image']
        #diff = im_out - torch.autograd.Variable(torch.from_numpy(target_im))
        diff = im_out - target_im
        loss = torch.mean(diff ** 2)

        if CUDA:
            im_out_ = im_out.cpu().data.numpy()
            loss_ = loss.cpu().data.numpy()
        else:
            im_out_ = im_out.data.numpy()
            loss_ = loss.data.numpy()

        #plt.imshow(im_out_)

        print('%d. loss= %f' % (iter, loss_))
        loss.backward()
        optimizer.step()

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    import copy
    scene = SCENE_BASIC
    #res = render_scene(scene, OUTPUT_FOLDER)
    input_scene = copy.deepcopy(SCENE_BASIC)
    input_scene['materials']['albedo'] = tch_var_f([
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.1, 0.8, 0.9],
        [0.1, 0.8, 0.9],
        [0.9, 0.1, 0.1],
    ])
    res = optimize_scene(input_scene, scene, OUTPUT_FOLDER, max_iter=4)
