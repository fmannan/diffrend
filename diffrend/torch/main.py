from diffrend.torch.params import OUTPUT_FOLDER, SCENE_BASIC
from diffrend.torch.renderer import Renderer
import matplotlib.pyplot as plt


def render_scene(scene, output_folder):
    # main render run
    res = Renderer().render(scene)

    im = res['image']

    plt.ion()
    plt.figure()
    plt.imshow(im)
    plt.title('Final Rendered Image')
    plt.savefig(output_folder + 'img_np.png')

    depth = res['depth']
    plt.figure()
    plt.imshow(depth)
    plt.title('Depth Image')
    plt.savefig(output_folder + 'img_depth_np.png')

    plt.ioff()
    plt.show()
    return res


if __name__ == '__main__':
    scene = SCENE_BASIC
    # from diffrend.numpy.scene import load_scene, load_mesh_from_file, obj_to_triangle_spec, mesh_to_scene
    #
    # mesh = load_mesh_from_file('../../data/bunny.obj')
    # tri_mesh = obj_to_triangle_spec(mesh)
    #
    # scene = mesh_to_scene(mesh)

    res = render_scene(scene, OUTPUT_FOLDER)
