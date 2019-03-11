import scenenet_pb2 as sn
import os
import json
import argparse


def main():
    """Generate light specs and obj file name"""
    parser = argparse.ArgumentParser(usage="python generate_scene_desc.py --pb=<path> --tmpl=<scene-template>")
    parser.add_argument('--tmpl', type=str, help='Scene template json file.')
    parser.add_argument('--pb', type=str, help='Path to the protobuf file.')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--data-root', type=str, default='./', help='Data root.')

    args = parser.parse_args()
    print(args)

    protobuf_path = os.path.expanduser(args.pb)
    data_root_path = os.path.expanduser(args.data_root)
    out_dir = os.path.expanduser(args.out)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    trajectories = sn.Trajectories()
    try:
        with open(protobuf_path, 'rb') as f:
            trajectories.ParseFromString(f.read())
    except IOError:
        print('Scenenet protobuf data not found at location:{0}'.format(data_root_path))

    print('Number of trajectories:{0}'.format(len(trajectories.trajectories)))

    for traj_no, traj in enumerate(trajectories.trajectories):
        layout_type = sn.SceneLayout.LayoutType.Name(traj.layout.layout_type)
        layout_path = traj.layout.model

        print('=' * 20)
        print('Render path:{0}'.format(traj.render_path))
        print('Layout type:{0} path:{1}'.format(layout_type, layout_path))
        print('=' * 20)
        obj_filename = 'trajectory_{}.obj'.format(traj_no)
        print(obj_filename)
        scene_desc = {"file-format": {"type": "diffrend", "version": "0.2"},
                      "glsl": {"vertex": "../shaders/phong/vs.glsl", "fragment": "../shaders/phong/fs.glsl"},
                      "camera": {
                          "proj_type": "perspective",
                          "viewport": [0, 0, 640, 480],
                          "fovy": 1.04,
                          "focal_length": 1.0,
                          "eye": [3.5, 2.0, 3.5, 1.0],
                          "up": [0.0, 1.0, 0.0, 0.0],
                          "at": [0.0, 1.0, 0.0, 1.0],
                          "near": 0.1, "far": 1000.0
                      },
                      "lights": {"pos": [],
                                 "color_idx": [],
                                 "attenuation": [],
                                 "ambient": [0.01, 0.01, 0.01]
                                 },
                      "colors": [],
                      "materials": {
                          "albedo": [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.5, 0.5, 0.5],
                                     [0.8, 0.8, 0.8], [0.44, 0.55, 0.64]],
                          "coeffs": [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                     [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
                      },
                      "objects": {"obj": [{"path": "../objs/scenenetrgbd/" + obj_filename, "material_idx": 4,
                                           "scale": [1.0, 1.0, 1.0]
                                           }
                                          ]
                                  },
                      "tonemap": {"type": "gamma", "gamma": [1.0]}
                      }
        for instance in traj.instances:
            if instance.instance_type == sn.Instance.LIGHT_OBJECT:
                light = instance.light_info
                light_type = sn.LightInfo.LightType.Name(light.light_type)
                light_intensity = [light.light_output.r, light.light_output.g, light.light_output.b]
                light_pos = [light.position.x, light.position.y, light.position.z]
                print('=' * 20)
                print('Light type:{0}'.format(light_type))
                print(light)
                print('emission: ', light_intensity)
                print('pos: ', light_pos)
                scene_desc["colors"].append(light_intensity)
                scene_desc['lights']["color_idx"].append(len(scene_desc["colors"]) - 1)
                scene_desc['lights']["pos"].append(light_pos)
                scene_desc['lights']["attenuation"].append([0.0, 0.0, 0.4])

        print(scene_desc)
        outfilename = 'scenenet_{:04}.json'.format(traj_no)
        print(outfilename)
        with open(os.path.join(out_dir, outfilename), 'w') as fid:
            json.dump(scene_desc, fid)


if __name__ == '__main__':
    main()