import scenenet_pb2 as sn
import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser(usage="read_protobuf")
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

    all_out_path = []
    for traj_no, traj in enumerate(trajectories.trajectories):
        layout_type = sn.SceneLayout.LayoutType.Name(traj.layout.layout_type)
        layout_path = traj.layout.model

        print('=' * 20)
        print('Render path:{0}'.format(traj.render_path))
        print('Layout type:{0} path:{1}'.format(layout_type, layout_path))
        print('=' * 20)

        '''
        The views attribute of trajectories contains all of the information
        about the rendered frames of a scene.  This includes camera poses,
        frame numbers and timestamps.
        '''
        camera_traj = []
        for view in traj.views:
            shutter_open = view.shutter_open
            camera = {'eye': [shutter_open.camera.x, shutter_open.camera.y, shutter_open.camera.z],
                      'at': [shutter_open.lookat.x, shutter_open.lookat.y, shutter_open.lookat.z],
                      'up': [0, 1, 0],  # default in scenenet.proto
                      'proj_type': "perspective",
                      'viewport': [0, 0, 640, 480],
                      'fovy': 1.04,
                      'focal_length': 1.0,
                      'frame_no': view.frame_num,
                      'timestamp': shutter_open.timestamp,
                      'obj_path': layout_path,
                      'render_path': traj.render_path,
                      "near": 0.1,
                      "far": 1000.0
                     }
            camera_traj.append(camera)
        outfilename = 'traj_{}_'.format(traj_no) + traj.render_path + '_' + layout_path
        outfilename = outfilename.replace('/', '_').replace('.obj', '.json')
        assert outfilename not in all_out_path
        print(outfilename)
        with open(os.path.join(out_dir, outfilename), 'w') as fid:
            json.dump(camera_traj, fid)
        print('')
        all_out_path.append(outfilename)


if __name__ == '__main__':
    main()