"""Animation generator."""
from diffrend.utils.sample_generator import uniform_sample_mesh
from diffrend.model import load_model
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np


def animate_sample_generation(model_name, obj=None, num_samples=1000,
                              out_dir=None, resample=True, rotate_angle=2):
    """Animation generator."""
    # Create the output folder
    if out_dir is not None:
        import os
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    # Load the model
    if obj is None:
        obj = load_model(model_name)

    # Create the window
    if out_dir is None:
        fig = plt.figure(figsize=(8.3, 8.3), dpi=72)
        ax = fig.add_subplot(111, projection='3d')
        plt.xlabel('x')
        plt.ylabel('y')

    # Find bounding box
    min_val = np.min(obj['v'])
    max_val = np.max(obj['v'])
    scale_dims = [min_val, max_val]
    # print(scale_dims)

    # Sample points from the mesh
    if not resample:
        pts_obj, vn = uniform_sample_mesh(obj, num_samples=num_samples)

    # Rotate the camera
    for angle in range(0, 360, rotate_angle):

        # Clean the window
        if out_dir is not None:
            # Redrawing with plt.save changes aspect ratio so had to recreate
            # everytime
            fig = plt.figure(figsize=(8.3, 8.3), dpi=72)
            ax = fig.add_subplot(111, projection='3d')
            plt.xlabel('x')
            plt.ylabel('y')
        else:
            ax.clear()

        # Sample points from the mesh
        if resample:
            pts_obj, vn = uniform_sample_mesh(obj, num_samples=num_samples)

        # Draw points
        print (pts_obj)
        ax.scatter(pts_obj[:, 0], pts_obj[:, 1], pts_obj[:, 2], s=1.3)
        ax.view_init(20, angle)
        ax.set_aspect('equal')
        ax.auto_scale_xyz(scale_dims, scale_dims, scale_dims)
        plt.draw()
        ax.apply_aspect()
        plt.tight_layout()
        plt.pause(.00001)

        # Save results
        if out_dir is not None:
            plt.savefig(out_dir + '/fig_%03d.png' % angle)
            plt.close(fig.number)


def main():
    import sys
    import argparse
    import argcomplete

    parser = argparse.ArgumentParser(
        description='Demo of generating uniform samples on a mesh.\n Usage: ' + sys.argv[0] +
                    '--obj ../../data/chair_0001.off [--out chair_anim]')
    parser.add_argument('--obj', type=str, help='Input mesh [obj|off|splat]', default='../../data/chair_0001.off')
    parser.add_argument('--out', type=str, help='Optional output directory for storing frames for animation.')
    parser.add_argument('--samples', type=int, help='Number of samples. default=1000', default=1000)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    print(args)
    plt.ion()
    animate_sample_generation(args.obj, num_samples=args.samples, out_dir=args.out)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
