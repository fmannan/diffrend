from diffrend.torch.params import SCENE_BASIC
from diffrend.torch.utils import tch_var_f, tch_var_l, get_data
from diffrend.torch.renderer import render
from diffrend.utils.sample_generator import uniform_sample_sphere
from diffrend.model import load_model, obj_to_triangle_spec
from data import DIR_DATA

import copy
import os
from time import time
import numpy as np
from imageio import imsave

import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
"""
save a batch of rendered images in npy
save in a different queue
N batches, M batchsize
usage:
python batch_render.py --config config_file.json

--scene <base_scene>
--model <model to be loaded>
--light_spec: how the lights would move
--cam_spec: how the camera would move
--num_batches
--batch_size
--ngpu
--out_dir

"""
WRITE_TO_FILE_QUEUE_SIZE = 10240


def save_to_file(out_dir, queue):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    while True:
        if not queue.empty():
            res = queue.get()
            if res is None:
                print("terminating")
                break
            suffix = res['suffix']
            im = np.uint8(255. * get_data(res['image']))
            depth = get_data(res['depth'])

            depth[depth >= res['camera_far']] = depth.min()
            im_depth = np.uint8(255. * (depth - depth.min()) / (depth.max() - depth.min()))

            imsave(out_dir + '/img' + suffix + '.png', im)
            imsave(out_dir + '/depth' + suffix + '.png', im_depth)


def batch_render_random_camera(filename, cam_dist, num_views, width, height,
                         fovy, focal_length, theta_range=None, phi_range=None,
                         axis=None, angle=None, cam_pos=None, cam_lookat=None,
                         double_sided=False, use_quartic=False, b_shadow=True,
                         tile_size=None, save_image_queue=None):
    rendering_time = []

    obj = load_model(filename)
    # normalize the vertices
    v = obj['v']
    axis_range = np.max(v, axis=0) - np.min(v, axis=0)
    v = (v - np.mean(v, axis=0)) / max(axis_range)  # Normalize to make the largest spread 1
    obj['v'] = v

    scene = copy.deepcopy(SCENE_BASIC)

    scene['camera']['viewport'] = [0, 0, width, height]
    scene['camera']['fovy'] = np.deg2rad(fovy)
    scene['camera']['focal_length'] = focal_length

    mesh = obj_to_triangle_spec(obj)
    faces = mesh['face']
    normals = mesh['normal']
    num_tri = faces.shape[0]

    if 'disk' in scene['objects']:
        del scene['objects']['disk']
    scene['objects'].update({'triangle': {'face': None, 'normal': None, 'material_idx': None}})
    scene['objects']['triangle']['face'] = tch_var_f(faces.tolist())
    scene['objects']['triangle']['normal'] = tch_var_f(normals.tolist())
    scene['objects']['triangle']['material_idx'] = tch_var_l(np.zeros(num_tri, dtype=int).tolist())

    scene['materials']['albedo'] = tch_var_f([[0.6, 0.6, 0.6]])
    scene['tonemap']['gamma'] = tch_var_f([1.0])  # Linear output

    # generate camera positions on a sphere
    if cam_pos is None:
        cam_pos = uniform_sample_sphere(radius=cam_dist, num_samples=num_views,
                                        axis=axis, angle=angle,
                                        theta_range=theta_range, phi_range=phi_range)
    lookat = cam_lookat if cam_lookat is not None else np.mean(v, axis=0)
    scene['camera']['at'] = tch_var_f(lookat)

    for idx in range(cam_pos.shape[0]):
        scene['camera']['eye'] = tch_var_f(cam_pos[idx])

        # main render run
        start_time = time()
        res = render(scene, tile_size=tile_size, tiled=tile_size is not None,
                     shadow=b_shadow, double_sided=double_sided,
                     use_quartic=use_quartic)
        res['suffix'] = '_{}'.format(idx)
        res['camera_far'] = scene['camera']['far']
        save_image_queue.put_nowait(get_data(res))
        rendering_time.append(time() - start_time)

    # Timing statistics
    print('Rendering time mean: {}s, std: {}s'.format(np.mean(rendering_time), np.std(rendering_time)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(usage="splat_gen_render_demo.py --model filename --out_dir output_dir "
                                           "--n 5000 --width 128 --height 128 --r 0.025 --cam_dist 5 --nv 10")
    parser.add_argument('--model', type=str, default=DIR_DATA + '/chair_0001.off', help='Path to the model file')
    parser.add_argument('--out_dir', type=str, default='./render_samples/', help='Directory for rendered images.')
    parser.add_argument('--width', type=int, default=128, help='Width of output image.')
    parser.add_argument('--height', type=int, default=128, help='Height of output image.')
    parser.add_argument('--cam_dist', type=float, default=5.0, help='Camera distance from the center of the object.')
    parser.add_argument('--nv', type=int, default=10, help='Number of views to generate.')
    parser.add_argument('--fovy', type=float, default=18.0, help='Field of view in the vertical direction.')
    parser.add_argument('--f', type=float, default=0.1, help='Focal length of camera.')
    parser.add_argument('--theta', nargs=2, type=float, help='Angle in degrees from the z-axis.')
    parser.add_argument('--phi', nargs=2, type=float, help='Angle in degrees from the x-axis.')
    parser.add_argument('--axis', nargs=3, type=float, help='Axis for random camera position.')
    parser.add_argument('--angle', type=float, help='Angular deviation from the mean axis.')
    parser.add_argument('--cam_pos', nargs=3, type=float, help='Camera position.')
    parser.add_argument('--at', nargs=3, type=float, help='Camera lookat position.')
    parser.add_argument('--double-sided', action='store_true', help='Render double-sided triangles.')
    parser.add_argument('--use-quartic', action='store_true', help='Use quartic attenuation.')
    parser.add_argument('--tile-size', type=int, default=64**2, help='tile size.')
    parser.add_argument('--shadow', action='store_true', default=True, help='Render shadows')

    args = parser.parse_args()
    print(args)

    axis = None
    angle = None
    theta = None
    phi = None
    if args.theta is None and args.phi is None and args.axis is None and args.angle is None:
        theta = [0, np.pi]
        phi = [0, 2 * np.pi]
    elif args.axis is not None and args.angle is not None:
        angle = np.deg2rad(args.angle)
        axis = args.axis
    else:
        theta = np.deg2rad(args.theta)
        phi = np.deg2rad(args.phi)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    cam_pos = args.cam_pos
    print(cam_pos)
    if cam_pos is not None:
        cam_pos = np.array([cam_pos])

    mp.set_start_method('spawn')
    save_image_queue = Queue(WRITE_TO_FILE_QUEUE_SIZE)
    write_to_disk_process = Process(target=save_to_file, args=(args.out_dir, save_image_queue))
    write_to_disk_process.start()

    batch_render_random_camera(filename=args.model, cam_dist=args.cam_dist, num_views=args.nv,
                               width=args.width, height=args.height,
                               fovy=args.fovy, focal_length=args.f,
                               theta_range=args.theta, phi_range=args.phi,
                               axis=axis, angle=angle, cam_pos=cam_pos, cam_lookat=args.at,
                               tile_size=args.tile_size, double_sided=args.double_sided,
                               b_shadow=args.shadow, use_quartic=args.use_quartic,
                               save_image_queue=save_image_queue)
    save_image_queue.put(None)
    write_to_disk_process.join()

