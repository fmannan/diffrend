import argparse
import json
import math
import os
import numpy as np

from diffrend.utils.sample_generator import uniform_sample_sphere
from diffrend.numpy.ops import normalize as np_normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=str, default='../scenes/basic_cube.json',
                        help='Path to the base scene json file')
    parser.add_argument('--out', type=str, default='./generated_scenes',
                        help='Directory to output the scene json files')
    parser.add_argument('-n', '--number', type=int, default=10000,
                        help='Number of random scenes to generate')
    parser.add_argument('--cam_dist', type=float, default=0.8,
                        help='Camera distance from the center of the object')
    parser.add_argument('--theta', nargs=2, type=float,
                        default=[20, 80], help='Range of angles (in degrees) from the z-axis.')
    parser.add_argument('--phi', nargs=2, type=float,
                        default=[20, 70], help='Range of angles (in degrees) from the x-axis.')
    parser.add_argument('--axis', nargs=3, default=[0, 1, 0], type=float, help='Axis for random object rotations.')

    args = parser.parse_args()

    # Create the output folder if it does not already exist
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # Load the template scene
    with open(args.template) as f:
        scene = json.load(f)

    for i in range(args.number):
        if (i + 1) % 100 == 0:
            print(f"Progress: {(i + 1) / args.number}%")
            
        # Generate a random camera eye position
        camera_pos = uniform_sample_sphere(radius=args.cam_dist, num_samples=1,
                                           theta_range=np.deg2rad(args.theta),
                                           phi_range=np.deg2rad(args.phi))[0]
        scene['camera']['eye'] = camera_pos.tolist()

        # Generate a random light position
        light_pos = uniform_sample_sphere(radius=args.cam_dist, num_samples=1,
                                          theta_range=np.deg2rad(args.theta),
                                          phi_range=np.deg2rad(args.phi))[0]
        # Only change the first light
        scene['lights']['pos'][0][:3] = light_pos.tolist()

        # Randomly rotate objects that are not the background:
        for obj in scene['objects']['obj']:
            # TODO this is pretty hacky, figure out a better way to do this
            if obj['path'] == "./objs/halfbox.obj":  # Ignore background objects
                continue

            random_axis = np_normalize(args.axis)
            random_angle = np.random.rand(1)[0] * np.pi * 2
            obj['rotate'] = dict(angle_deg=random_angle,
                                 axis=random_axis.tolist())

        # Output the generated scene json file
        with open(f'{args.out}/scene{str(i).zfill(math.ceil(math.log10(args.number)))}.json', 'w') as outfile:
            json.dump(scene, outfile)


if __name__ == '__main__':
    main()
