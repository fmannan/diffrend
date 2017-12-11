"""Shapenet dataset loader."""
import os
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
from diffrend.model import load_model, obj_to_splat
from analysis.gen_surf_pts.generator_anim import animate_sample_generation
from diffrend.generator.splats import SplatScene
from diffrend.torch.renderer import render
import matplotlib.pyplot as plt

# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")
# plt.ion()   # interactive mode


class ShapeNetDataset(Dataset):
    """Shapenet dataset."""

    def __init__(self, root_dir, synsets="all", classes=None, transform=None):
        """Constructor.

        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.classes = classes
        self.synsets = synsets
        self.root_dir = root_dir
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

    def __len__(self):
        """Get dataset length."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get item."""
        # Get object path
        synset, obj = self.samples[idx]
        obj_path = os.path.join(self.root_dir, synset, obj, 'models',
                                'model_normalized.obj')

        # Load obj model
        obj_model = load_model(obj_path)

        # Show loaded model
        animate_sample_generation(model_name=None, obj=obj_model,
                                  num_samples=10, out_dir=None,
                                  resample=False, rotate_angle=360)

        # Convert model to splats
        splats_model = obj_to_splat(obj_model, use_circum_circle=True)

        # Create a splat scene that can be redenred
        n_splats = splats_model['vn'].shape[0]
        splat_scene = SplatScene(n_lights=2, n_splats=n_splats)

        # Add the splats to the scene
        for i, splat in enumerate(np.column_stack((
                splats_model['vn'], splats_model['v'],
                np.asarray(splats_model['r'], dtype=np.float32),
                np.ones((n_splats, 3), dtype=np.float32)))):
            splat_scene.set_splat_array(i, splat)

        # Camera
        splat_scene.set_camera(
            viewport=np.asarray([0, 0, 64, 64], dtype=np.float32),
            eye=np.asarray([0.0, 1.0, 10.0, 1.0], dtype=np.float32),
            up=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            at=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            fovy=90.0, focal_length=1.0, near=1.0, far=1000.0)

        # Tonemap
        splat_scene.set_tonemap(tonemap_type='gamma', gamma=0.8)

        # Lights
        splat_scene.set_light(
            id=0,
            pos=np.asarray([20., 20., 20.], dtype=np.float32),
            color=np.asarray([0.8, 0.1, 0.1], dtype=np.float32),
            attenuation=np.asarray([0.2, 0.2, 0.2], dtype=np.float32))

        splat_scene.set_light(
            id=1,
            pos=np.asarray([-15, 3., 15.], dtype=np.float32),
            color=np.asarray([0.8, 0.1, 0.1], dtype=np.float32),
            attenuation=np.asarray([0., 1., 0.], dtype=np.float32))

        # print (splat_scene.scene)
        # print (splat_scene.to_pytorch())

        # import ipdb; ipdb.set_trace()
        # print (splat_scene.to_pytorch())

        # Show splats model
        res = render(splat_scene.to_pytorch())
        print ('Finished render')

        if True:
            im = res['image'].cpu().data.numpy()
        else:
            im = res['image'].data.numpy()

        plt.ion()
        plt.figure()
        plt.imshow(im)
        plt.title('Final Rendered Image')
        plt.show()
        print ("Finish plot")
        exit()

        # Add model and synset to the output dictionary
        sample = {'splats': splats_model, 'synset': synset}

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
        all_synsets = [f for f in os.listdir(self.root_dir)
                       if os.path.isdir(os.path.join(self.root_dir, f))]

        # Read the taxonomy metadata file
        with open(os.path.join(self.root_dir, "taxonomy.json")) as json_file:
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
            synset_path = os.path.join(self.root_dir, synset)
            for o in os.listdir(synset_path):
                self.samples.append([synset, o])


def main():
    """Test function."""
    dataset = ShapeNetDataset(
        # root_dir='/mnt/AIDATA/home/dvazquez/datasets/shapenet/ShapeNetCore.v2',
        root_dir='/home/dvazquez/datasets/shapenet/ShapeNetCore.v2',
        synsets=None, classes=["airplane", "microphone"], transform=None)
    print (len(dataset))

    for f in dataset:
        print (f)


if __name__ == "__main__":
    main()
