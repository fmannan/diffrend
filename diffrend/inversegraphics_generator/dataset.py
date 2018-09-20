import os
import time

import numpy as np
import tempfile

from diffrend.inversegraphics_generator.obj_generator import ObjGenerator
from diffrend.inversegraphics_generator.iqtest_objs import get_data_dir

MAX_GRID = 8

"""
fmannan: Refactoring this to make it self contained. The caller doesn't have to
worry about doing cleanup and buffer. The caller only does
for sample in iq.get_training_samples_unordered(100):
    .... use the sample
or for the QA set:
for sample in iq.get_training_questions_answers(100):
    .... use the sample

@Florian: self.train[0, 3] is empty !
"""
class IqDataset(object):
    def __init__(self, path, cleanup_interval=100):
        ds = np.load(path)
        self.train = ds["train"]
        self.test = ds["test"]
        self.val = ds["val"]
        print("dataset loaded:", path)
        self.og = ObjGenerator(MAX_GRID, 1.0)
        self.train_idx_unord = 0
        self.train_idx_qa = 0

        self.cleanup_interval = cleanup_interval
        self.tmp_training_path = None
        self.tmp_qa_path = None
        self.tmp_training_folder_size = 0
        self.tmp_qa_folder_size = 0

    def _grid_to_file(self, grid, folder, idx, filepattern="{:06d}.obj"):
        v, f = self.og.grid_to_cubes(grid)
        out_path = os.path.join(folder.name, filepattern.format(idx))
        self.og.write_obj(grid, v, f, out_path)
        return out_path

    def _grid_to_obj(self, grid):
        v, f = self.og.grid_to_cubes(grid)
        if len(v) == 0 or len(f) == 0:
            print("ERROR: grid is all 0")
            #raise RuntimeError("ERROR: grid is all 0")
        return self.og.generate_obj(grid, v, f)

    def get_training_samples_unordered(self, n=1):
        if self.tmp_training_path is None:
            # make tmp dir
            self.tmp_training_path = tempfile.TemporaryDirectory()
            self.tmp_training_folder_size = 0
        # get N samples and write to disk
        for idx in range(n):
            self.tmp_training_folder_size += 1
            self.train_idx_unord += 1
            if self.train_idx_unord == len(self.train):
                self.train_idx_unord = 0

            obj_path = self._grid_to_file(self.train[self.train_idx_unord, 0],
                                          self.tmp_training_path, self.train_idx_unord)
            yield {'ref': obj_path}

        # cleanup once reached the cleanup interval
        # avoids doing this in every iteration to save I/O cost. But can also be made async.
        if self.tmp_training_folder_size >= self.cleanup_interval:
            self.tmp_training_path.cleanup()
            self.tmp_training_folder_size = 0

    def get_training_questions_answers(self, n=1):
        if self.tmp_qa_path is None:
            # make tmp dir
            self.tmp_qa_path = tempfile.TemporaryDirectory()
            self.tmp_qa_folder_size = 0

        # get N samples and write to disk
        for idx in range(n):
            self.tmp_qa_folder_size += 1
            self.train_idx_qa += 1
            if self.train_idx_qa == len(self.train):
                self.train_idx_qa = 0

            ref_path = self._grid_to_file(self.train[self.train_idx_qa, 0], self.tmp_qa_path,
                                          self.train_idx_qa, "{:06d}-ref.obj")
            ans1_path = self._grid_to_file(self.train[self.train_idx_qa, 1], self.tmp_qa_path,
                                           self.train_idx_qa, "{:06d}-ans1.obj")
            ans2_path = self._grid_to_file(self.train[self.train_idx_qa, 2], self.tmp_qa_path,
                                           self.train_idx_qa, "{:06d}-ans2.obj")
            ans3_path = self._grid_to_file(self.train[self.train_idx_qa, 3], self.tmp_qa_path,
                                           self.train_idx_qa, "{:06d}-ans3.obj")
            yield {'ref': ref_path, 'ans1': ans1_path, 'ans2': ans2_path, 'ans3': ans3_path}

        # cleanup once reached the cleanup interval
        # avoids doing this in every iteration to save I/O cost. But can also be made async.
        if self.tmp_qa_folder_size >= self.cleanup_interval:
            self.tmp_qa_path.cleanup()
            self.tmp_qa_folder_size = 0

    def get_training_samples_unordered_inmem(self, n=1):
        # get N samples and write to disk
        for idx in range(n):
            self.train_idx_unord += 1
            if self.train_idx_unord == len(self.train):
                self.train_idx_unord = 0

            obj = self._grid_to_obj(self.train[self.train_idx_unord, 0])

            yield {'ref': obj}

    def get_training_questions_answers_inmem(self, n=1):
        # get N samples and write to disk
        for idx in range(n):
            self.train_idx_qa += 1
            if self.train_idx_qa == len(self.train):
                self.train_idx_qa = 0

            ref = self._grid_to_obj(self.train[self.train_idx_qa, 0])
            ans1 = self._grid_to_obj(self.train[self.train_idx_qa, 1])
            ans2 = self._grid_to_obj(self.train[self.train_idx_qa, 2])
            ans3 = self._grid_to_obj(self.train[self.train_idx_qa, 3])

            yield {'ref': ref, 'ans1': ans1, 'ans2': ans2, 'ans3': ans3}


if __name__ == '__main__':
    # iq = IqDataset(os.path.expanduser("~/data/ig/iqtest-v1.npz"))
    iq = IqDataset(os.path.join(get_data_dir(), "iqtest-v1.npz"))
    print(iq.train.shape)
    print(iq.test.shape)
    print(iq.val.shape)

    # print(iq.get_training_samples_unordered(5))
    # print(iq.get_training_questions_answers(5))
    #
    # time.sleep(30) # use this time to do `ls` on the directories printed above
    # iq.cleanup()
    for sample in iq.get_training_samples_unordered(10):
        print(sample)
    for qa_sample in iq.get_training_questions_answers(5):
        print(qa_sample)

