import os
import time

import numpy as np
import tempfile

from diffrend.inversegraphics_generator.obj_generator import ObjGenerator
from diffrend.inversegraphics_generator.iqtest_objs import get_data_dir

MAX_GRID = 8


class IqDataset(object):
    def __init__(self, path):
        ds = np.load(path)
        self.train = ds["train"]
        self.test = ds["test"]
        self.val = ds["val"]
        print("dataset loaded:", path)
        self.og = ObjGenerator(MAX_GRID, 1.0)
        self.train_idx_unord = 0
        self.train_idx_qa = 0
        self.paths = []
        # self.tmp = tempfile.TemporaryDirectory()

    def _grid_to_file(self, grid, folder, idx, filepattern="{:06d}.obj"):
        v, f = self.og.grid_to_cubes(grid)
        out_path = os.path.join(folder.name, filepattern.format(idx))
        self.og.write_obj(grid, v, f, out_path)

    def get_training_samples_unordered(self, n=1):
        # make tmp dir
        folder = tempfile.TemporaryDirectory()
        self.paths.append(folder)

        # get N samples and write to disk
        for idx in range(n):
            self.train_idx_unord += 1
            if self.train_idx_unord == len(self.train):
                self.train_idx_unord = 0

            self._grid_to_file(self.train[self.train_idx_unord, 0], folder, self.train_idx_unord)

        # return path
        return folder.name

    def get_training_questions_answers(self, n=1):
        # make tmp dir
        folder = tempfile.TemporaryDirectory()
        self.paths.append(folder)

        # get N samples and write to disk
        for idx in range(n):
            self.train_idx_qa += 1
            if self.train_idx_qa == len(self.train):
                self.train_idx_qa = 0

            self._grid_to_file(self.train[self.train_idx_qa, 0], folder, self.train_idx_qa, "{:06d}-ref.obj")
            self._grid_to_file(self.train[self.train_idx_qa, 1], folder, self.train_idx_qa, "{:06d}-ans1.obj")
            self._grid_to_file(self.train[self.train_idx_qa, 2], folder, self.train_idx_qa, "{:06d}-ans2.obj")
            self._grid_to_file(self.train[self.train_idx_qa, 3], folder, self.train_idx_qa, "{:06d}-ans3.obj")

        # return path
        return folder.name

    def cleanup(self):
        for tmp in self.paths:
            tmp.cleanup()


if __name__ == '__main__':
    # iq = IqDataset(os.path.expanduser("~/data/ig/iqtest-v1.npz"))
    iq = IqDataset(os.path.join(get_data_dir(),"iqtest-v1.npz"))
    print(iq.train.shape)
    print(iq.test.shape)
    print(iq.val.shape)

    print(iq.get_training_samples_unordered(5))
    print(iq.get_training_questions_answers(5))

    time.sleep(30) # use this time to do `ls` on the directories printed above
    iq.cleanup()
