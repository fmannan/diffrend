"""Shapenet dataset."""
import os
import numpy as np
from torch.utils.data import Dataset
from diffrend.model import load_model, obj_to_triangle_spec
from diffrend.utils.sample_generator import uniform_sample_mesh
from diffrend.numpy.ops import axis_angle_matrix
from diffrend.numpy.ops import normalize as np_normalize


class ObjectsFolderMultiObjectDataset(Dataset):
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
        self.loaded = False
        self.bg_obj = None
        self.fg_obj = None
        # Get object paths

        # Get object paths
        self._get_objects_paths()
        print ("Total samples: {}".format(len(self.samples)))

        # self.scene = self._create_scene()

    def __len__(self):
        """Get dataset length."""
        return len(self.samples)
    def load_sample(self, obj_path):
        # if not self.loaded:
        #     self.fg_obj = load_model(obj_path)
        #     self.bg_obj = load_model(self.opt.bg_model)
        #     self.loaded = True
        obj_model = load_model(obj_path)  #'../../../data/sphere_halfbox_v2.obj'
        obj2 = load_model(self.opt.bg_model)
        # obj_model = self.fg_obj
        # obj2 = self.bg_obj
        #v1 = (obj_model['v'] - obj_model['v'].mean()) / (obj_model['v'].max() - obj_model['v'].min())
        v2 = obj2['v']  # / (obj2['v'].max() - obj2['v'].min())
        scale = (obj2['v'].max() - obj2['v'].min()) * 0.3
        offset = np.array([14, 13, 15.0]) #+ 2 * np.abs(np.random.randn(3))
        if self.opt.only_background:
            v=v2
            f=obj2['f']
        elif self.opt.only_foreground:
            v=obj_model['v']

            f=obj_model['f']
            if self.opt.random_rotation:
                random_axis = np_normalize(self.opt.axis)
                random_angle = np.random.rand(1) * np.pi * 2
                M = axis_angle_matrix(axis=random_axis, angle=random_angle)

                #v1 = v1 - np.mean(v1, axis=0)
                v = np.matmul(v, M.transpose(1, 0)[:3, :3])
            v1=v
        else:
            if self.opt.random_rotation:
                random_axis = np_normalize(self.opt.axis)
                random_angle = np.random.rand(1) * np.pi * 2
                M = axis_angle_matrix(axis=random_axis, angle=random_angle)
                M[:3, 3] = offset
                v1 = np.matmul(scale * v1, M.transpose(1, 0)[:3, :3]) + M[:3, 3]
            else:
                v1 = scale * v1 + offset
            v = np.concatenate((v1, v2))
            f = np.concatenate((obj_model['f'], obj2['f'] + v1.shape[0]))

        obj_model = {'v': v, 'f': f}

        if self.opt.use_mesh:
            # normalize the vertices


            v = obj_model['v']

            axis_range = np.max(v, axis=0) - np.min(v, axis=0)
            object_center= (np.mean(v1, 0) - np.mean(v, 0))/ max(axis_range)
            v = (v - np.mean(v, axis=0)) / max(axis_range)  # Normalize to make the largest spread 1
            #v =(v - [2, 2, 2])/4

            obj_model['v'] = v
            mesh = obj_to_triangle_spec(obj_model)
            meshes = {'face': mesh['face'].astype(np.float32),
                      'normal': mesh['normal'].astype(np.float32),
                      'object_center': (object_center).astype(np.float32)}
            sample = {'synset': 0, 'mesh': meshes}
        else:
            # Sample points from the 3D mesh
            v, vn = uniform_sample_mesh(obj_model,
                                        num_samples=self.opt.n_splats)
            # Normalize the vertices
            v = (v - np.mean(v2, axis=0)) / (v.max() - v.min())

            # Save the splats
            splats = {'pos': v.astype(np.float32),
                      'normal': vn.astype(np.float32)}

            # Add model and synset to the output dictionary
            sample = {'synset': 0, 'splats': splats}

        # Transform
        if self.transform:
            sample = self.transform(sample)

        # use obj_path to determine uniqueness
        sample['obj_path'] = obj_path
        return sample

    #def __getitem__(self, idx):
    def get_sample(self):
        """Get item."""
        # Get object path
        samples = []

        print(self.opt.root_dir)
        
        if(self.opt.isSupervised):
            pass
            #for sample_name in []:
            #    obj_path = os.path.join(self.opt.root_dir, sample_name)
            #    samples.append(self.load_sample(obj_path))
        else:
            obj_path = os.path.join(self.opt.root_dir, self.samples[idx])
            samples.append(self.load_sample(obj_path))
        return samples

    def _get_objects_paths(self,):
        print (self.opt.root_dir)
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
