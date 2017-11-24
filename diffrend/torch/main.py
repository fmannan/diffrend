from diffrend.torch.params import OUTPUT_FOLDER, SCENE_BASIC
from diffrend.torch.renderer import render
import numpy as np
import matplotlib.pyplot as plt
import os


def render_scene(scene, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # main render run
    res = render(scene)

    im = res['image'].data.numpy()

    plt.ion()
    plt.figure()
    plt.imshow(im)
    plt.title('Final Rendered Image')
    plt.savefig(output_folder + 'img_torch.png')

    depth = res['depth'].data.numpy()
    depth[depth >= scene['camera']['far']] = np.inf
    plt.figure()
    plt.imshow(depth)
    plt.title('Depth Image')
    plt.savefig(output_folder + 'img_depth_torch.png')

    plt.ioff()
    plt.show()
    return res


if __name__ == '__main__':
    scene = SCENE_BASIC
    res = render_scene(scene, OUTPUT_FOLDER)
