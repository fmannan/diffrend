# from __future__ import print_function, division
import os
# import torch
# import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
from diffrend.model import load_model, obj_to_splat
from analysis.gen_surf_pts.generator_anim import animate_sample_generation


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
                                  num_samples=1000, out_dir=None,
                                  resample=False, rotate_angle=360)

        # Convert model to splats
        splats_model = obj_to_splat(obj_model, use_circum_circle=True)

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
