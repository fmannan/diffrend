"""Sapenet dataset."""
import os
import numpy as np
from torch.utils.data import Dataset
from diffrend.model import load_model, obj_to_triangle_spec
from diffrend.utils.sample_generator import uniform_sample_mesh


class ObjectsFolderDataset(Dataset):
    """Objects folder dataset."""

    def __init__(self, opt, transform=None):
        """Constructor.

        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.opt = opt
        self.transform = transform
        self.n_samples = 0
        self.samples = []

        # Get object paths
        self._get_objects_paths()
        print ("Total samples: {}".format(len(self.samples)))

        # self.scene = self._create_scene()

    def __len__(self):
        """Get dataset length."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get item."""
        # Get object path
        obj_path = os.path.join(self.opt.root_dir, self.samples[idx])
        print (obj_path)

        # Load obj model
        obj_model = load_model(obj_path)

        if self.opt.use_mesh:
            # normalize the vertices
            v = obj_model['v']
            axis_range = np.max(v, axis=0) - np.min(v, axis=0)
            v = (v - np.mean(v, axis=0)) / max(axis_range)  # Normalize to make the largest spread 1
            obj_model['v'] = v
            mesh = obj_to_triangle_spec(obj_model)
            meshes = {'face': mesh['face'].astype(np.float32),
                      'normal': mesh['normal'].astype(np.float32)}
            sample = {'synset': 0, 'mesh': meshes}
        else:
            # Sample points from the 3D mesh
            v, vn = uniform_sample_mesh(obj_model,
                                        num_samples=self.opt.n_splats)
            # Normalize the vertices
            v = (v - np.mean(v, axis=0)) / (v.max() - v.min())

            # Save the splats
            splats = {'pos': v.astype(np.float32),
                      'normal': vn.astype(np.float32)}

            # Add model and synset to the output dictionary
            sample = {'synset': 0, 'splats': splats}

        # Transform
        if self.transform:
            sample = self.transform(sample)

        return sample

    def _get_objects_paths(self,):
        for o in os.listdir(self.opt.root_dir):
            self.samples.append(o)


def main():
    """Test function."""
    dataset = ObjectsFolderDataset(
        root_dir='/home/dvazquez/datasets/shapenet/ShapeNetCore.v2',
        transform=None)
    print (len(dataset))

    for f in dataset:
        print (f)


if __name__ == "__main__":
    main()
