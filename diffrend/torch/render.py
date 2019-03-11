from diffrend.torch.utils import tch_var_f, tch_var_l, get_data
from diffrend.torch.renderer import render
from diffrend.model import load_obj, obj_to_triangle_spec
import numpy as np


def load_scene(scene_filename):
    """Loads a diffrend scene file
    Args:
        fname:

    Returns:
        scene
    """
    import json
    import os

    with open(scene_filename, 'r') as fid:
        scene = json.load(fid)

    basedir = os.path.dirname(scene_filename)
    objects = scene['objects']['obj']
    mesh = {'face': None,
            'normal': None,
            'material_idx': None
           }
    for obj in objects:
        print(obj)
        model_path = os.path.join(basedir, obj['path'])
        print(os.path.exists(model_path))
        obj_model = load_obj(model_path)
        meshes = obj_to_triangle_spec(obj_model)
        material_idx = np.ones(meshes['face'].shape[0]) * obj['material_idx']
        if mesh['face'] is None:
            mesh['face'] = meshes['face']
            mesh['normal'] = meshes['normal']
            mesh['material_idx'] = material_idx
        else:
            mesh['face'] = np.concatenate((mesh['face'], meshes['face']))
            mesh['normal'] = np.concatenate((mesh['normal'], meshes['normal']))
            mesh['material_idx'] = np.concatenate((mesh['material_idx'], material_idx))
    scene['objects']['triangle'] = mesh
    del scene['objects']['obj']
    return scene


def make_torch_tensor(var):
    var_elem = var
    if type(var) is list:
        var_elem = var[0]
    return tch_var_l(var) if type(var_elem) is int else tch_var_f(var)


def make_torch_var(var_dict):
    for k in var_dict:
        var = var_dict[k]
        if type(var) is dict:
            res = make_torch_var(var)
        elif type(var) is list:
            res = make_torch_tensor(var)
        elif type(var) is np.ndarray:
            res = make_torch_tensor(var.tolist())
        else:
            res = var
        var_dict[k] = res
    return var_dict


def main():
    import argparse
    parser = argparse.ArgumentParser(usage="render.py --scene scene_filename --out_dir output_dir")
    parser.add_argument('--scene', type=str, default='../../scenes/basic_triangle_phong.json', help='Path to the model file')
    parser.add_argument('--out_dir', type=str, default='./render_samples/', help='Directory for rendered images.')
    args = parser.parse_args()
    print(args)

    scene = load_scene(args.scene)

    scene = make_torch_var(scene)
    res = render(scene)
    img = res['image'].cpu().numpy().squeeze()
    depth = res['depth'].cpu().numpy().squeeze()

    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure()
    plt.imshow(img)

    plt.figure()
    plt.imshow(depth)


if __name__ == '__main__':
    main()

