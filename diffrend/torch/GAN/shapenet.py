import os
import numpy as np
import copy
import json
import torch
from torch.utils.data import Dataset
# from torchvision import transforms, utils
from diffrend.model import load_model
from diffrend.torch.params import SCENE_BASIC
from diffrend.torch.utils import tch_var_f, tch_var_l, where
# from analysis.gen_surf_pts.generator_anim import animate_sample_generation
from diffrend.torch.renderer import render
from diffrend.utils.sample_generator import (uniform_sample_mesh,
                                             uniform_sample_sphere)


class ShapeNetDataset(Dataset):
    """Shapenet dataset."""

    def __init__(self, opt, transform=None):
        """Constructor.

        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.opt = opt
        if opt.synsets == '':
            self.synsets = None
        else:
            self.synsets = opt.synsets.split(',')
        if opt.classes == '':
            self.classes = None
        else:
            self.classes = opt.classes.split(',')
        self.transform = transform
        self.n_samples = 0
        self.samples = []

        # Get taxonomy dictionaries
        self._get_taxonomy()

        # Check the selected synsets/classes
        self._check_synsets_classes()
        print ("Selected synsets: {}".format(self.synsets))
        print ("Selected classes: {}".format(self.classes))

        # Get object paths
        self._get_objects_paths()
        print ("Total samples: {}".format(len(self.samples)))

        # Create semi-empty scene
        self._create_scene()

    def __len__(self):
        """Get dataset length."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get item."""
        # Get object path
        synset, obj = self.samples[idx]
        obj_path = os.path.join(self.opt.root_dir, synset, obj, 'models',
                                'model_normalized.obj')

        # Load obj model
        obj_model = load_model(obj_path)

        # Show loaded model
        # animate_sample_generation(model_name=None, obj=obj_model,
        #                           num_samples=1000, out_dir=None,
        #                           resample=False, rotate_angle=360)

        # Convert model to splats and render views
        samples = self._generate_samples(obj_model)

        # Add model and synset to the output dictionary
        sample = {'obj': obj_model, 'samples': samples, 'synset': synset}

        # Transform
        if self.transform:
            sample = self.transform(sample)

        return sample

    def _get_taxonomy(self,):
        """Read json metadata file."""
        # Create the output dictionaries
        self.synset_to_class = {}
        self.class_to_synset = {}

        # Get the list of all the possible synsets
        all_synsets = [f for f in os.listdir(self.opt.root_dir)
                       if os.path.isdir(os.path.join(self.opt.root_dir, f))]

        # Read the taxonomy metadata file
        with open(os.path.join(self.opt.root_dir,
                               "taxonomy.json")) as json_file:
            json_data = json.load(json_file)

        # Parse the json data looking for basic synsets
        for el in json_data:
            if "synsetId" in el and "name" in el:
                if el["synsetId"] in all_synsets:
                    synset = str(el["synsetId"])
                    name = str(el["name"]).split(',')[0]
                    self.synset_to_class[synset] = name
                    self.class_to_synset[name] = synset

    def _check_synsets_classes(self,):
        # Check selected classes/synsets
        if self.classes is None and self.synsets is None:
            raise ValueError("Select classes to load")
        if self.classes is not None and self.synsets is not None:
            raise ValueError("Select or synsets or classes")

        # Check selected synsets
        if self.synsets == 'all':
            self.synsets = [k for k, v in self.synset_to_class.items()]
        elif self.synsets is not None:
            for el in self.synsets:
                if el not in self.synset_to_class:
                    raise ValueError("Unknown synset: " + el)

        # Check selected classes
        if self.classes == 'all':
            self.classes = [v for k, v in self.synset_to_class.items()]
        elif self.classes is not None:
            self.synsets = []
            for el in self.classes:
                if el not in self.class_to_synset:
                    raise ValueError("Unknown class: " + el)
                else:
                    self.synsets.append(self.class_to_synset[el])

    def _get_objects_paths(self,):
        for synset in self.synsets:
            synset_path = os.path.join(self.opt.root_dir, synset)
            for o in os.listdir(synset_path):
                self.samples.append([synset, o])

    def _create_scene(self,):
        """Create a semi-empty scene with camera parameters."""
        # Create a splats rendering scene
        self.scene = copy.deepcopy(SCENE_BASIC)

        # Define the camera parameters
        self.scene['camera']['viewport'] = [0, 0, self.opt.width,
                                            self.opt.height]
        self.scene['camera']['fovy'] = np.deg2rad(self.opt.fovy)
        self.scene['camera']['focal_length'] = self.opt.focal_length
        self.scene['objects']['disk']['radius'] = tch_var_f(
            np.ones(self.opt.n_splats) * self.opt.splats_radius)
        self.scene['objects']['disk']['material_idx'] = tch_var_l(
            np.zeros(self.opt.n_splats, dtype=int).tolist())
        self.scene['materials']['albedo'] = tch_var_f([[0.6, 0.6, 0.6]])
        self.scene['tonemap']['gamma'] = tch_var_f([1.0])  # Linear output

    def set_camera_pos(self, cam_dist=None, cam_pos=None):
        """Set camera pose."""
        # Check camera Parameters
        if cam_dist is None and cam_pos is None:
            raise ValueError('Use parameter cam_dist or cam_pos')
        elif cam_dist is not None and cam_pos is not None:
            raise ValueError('Use parameter cam_dist or cam_pos. No both.')
        elif cam_dist is not None:
            self.single_view = False
            self.cam_pos = uniform_sample_sphere(
                radius=cam_dist, num_samples=self.opt.batchSize)
        else:
            self.single_view = True
            self.cam_pos = cam_pos

    def _generate_samples(self, obj, verbose=False):
        """Generate random samples of an object from the same camera position.

        Randomly generate N samples on a surface and render them. The samples
        include position and normal, the radius is set to a constant.
        """
        data = []
        for idx in range(self.opt.batchSize):
            # Sample points from the 3D mesh
            v, vn = uniform_sample_mesh(obj, num_samples=self.opt.n_splats)

            # Normalize the vertices
            v = (v - np.mean(v, axis=0)) / (v.max() - v.min())

            # Save the splats into the rendering scene
            self.scene['objects']['disk']['pos'] = tch_var_f(v)
            self.scene['objects']['disk']['normal'] = tch_var_f(vn)

            # Set camera position
            if self.single_view:
                self.scene['camera']['eye'] = tch_var_f(self.cam_pos)
            else:
                self.scene['camera']['eye'] = tch_var_f(self.cam_pos[idx])

            # Render scene
            res = render(self.scene)
            depth = res['depth']
            # im = res['image']

            # Normalize depth image
            cond = depth >= self.scene['camera']['far']
            depth = where(cond, torch.min(depth), depth)
            im_depth = ((depth - torch.min(depth)) /
                        (torch.max(depth) - torch.min(depth)))

            # Add depth image to the output structure
            data.append(im_depth.unsqueeze(0))

        return torch.stack(data)


def main():
    """Test function."""
    dataset = ShapeNetDataset(
        root_dir='/home/dvazquez/datasets/shapenet/ShapeNetCore.v2',
        synsets=None, classes=["airplane", "microphone"], transform=None)
    print (len(dataset))

    for f in dataset:
        print (f)


if __name__ == "__main__":
    main()
