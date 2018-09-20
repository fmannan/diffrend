import errno
import io
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # don't remove this import

from diffrend.inversegraphics_generator.constants import actions


class ObjGenerator(object):

    def __init__(self, grid_size, cube_len=1.0, deadlock_safe=100):
        self.grid_size = grid_size
        self.cube_len = cube_len
        self.deadlock_safe = deadlock_safe

    def grid_to_str(self, grid):
        out = grid.flatten()
        out = "".join([str(x) for x in out])
        return out

    def str_to_grid(self, string, replace_head=False):
        out = [int(x) for x in string]
        out = np.array(out, dtype=np.uint8).reshape((self.grid_size, self.grid_size, self.grid_size))
        if replace_head:
            out[out == 2] = 1
        return out

    def step_is_valid(self, grid, pos):
        # check for boundaries:
        if pos[0] < 0 or pos[1] < 0 or pos[2] < 0:
            return False
        if pos[0] > self.grid_size - 1 or \
                pos[1] > self.grid_size - 1 or \
                pos[2] > self.grid_size - 1:
            return False
        if grid[pos[0], pos[1], pos[2]] == 1:
            return False

        return True

    @staticmethod
    def move_cube(vertices, x, y, z, faces, face_offset):
        vertices_out = []
        for v in vertices:
            vertices_out.append((v[0] + x, v[1] + y, v[2] + z))

        faces_out = []
        for f in faces:
            faces_out.append((f[0] + face_offset,
                              f[1] + face_offset,
                              f[2] + face_offset))

        return vertices_out, faces_out

    def make_cube(self):
        # create the cube at (0,0,0) then move later

        vertices = []
        faces = []

        # all 4 corners:
        vertices.append((0, 0, 0))  # 0

        vertices.append((self.cube_len, 0, 0))  # 1
        vertices.append((0, self.cube_len, 0))  # 2
        vertices.append((0, 0, self.cube_len))  # 3

        vertices.append((0, self.cube_len, self.cube_len))  # 4
        vertices.append((self.cube_len, 0, self.cube_len))  # 5
        vertices.append((self.cube_len, self.cube_len, 0))  # 6

        vertices.append((self.cube_len, self.cube_len, self.cube_len))  # 7

        ### viewer standing on x axis
        ## bottom
        faces.append((0, 1, 6))
        faces.append((0, 2, 6))

        ## top
        faces.append((3, 4, 7))
        faces.append((3, 5, 7))

        ## back
        faces.append((0, 2, 4))
        faces.append((0, 3, 4))

        ## front
        faces.append((1, 5, 7))
        faces.append((1, 6, 7))

        ## left
        faces.append((0, 1, 5))
        faces.append((0, 3, 5))

        ## right
        faces.append((2, 4, 7))
        faces.append((2, 6, 7))

        return vertices, faces

    def plot_cube(self, vertices, faces):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for f in faces:
            for edge in [(f[0], f[1]),
                         (f[1], f[2]),
                         (f[2], f[0])]:
                ax.plot(
                    [vertices[edge[0]][0], vertices[edge[1]][0]],
                    [vertices[edge[0]][1], vertices[edge[1]][1]],
                    [vertices[edge[0]][2], vertices[edge[1]][2]],
                    'ro-'
                )

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_zlim(0, self.grid_size)

    def walk_snek(self, snek_len_min, snek_len_max):
        # make grid
        grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), np.uint8)

        # sample snek size
        snek_len = np.random.randint(snek_len_min, snek_len_max)

        # init
        start = (
            np.random.randint(0, self.grid_size),
            np.random.randint(0, self.grid_size),
            np.random.randint(0, self.grid_size)
        )
        grid[start[0], start[1], start[2]] = 1

        last_pos = start

        # walk until snek is long enough
        for step in range(snek_len - 1):

            # try n find possible move
            deadlock_counter = 0
            while True:
                step = random.sample(actions, 1)[0]
                tmp_pos = np.array(last_pos) + np.array(step)
                if self.step_is_valid(grid, tmp_pos):
                    break

                deadlock_counter += 1
                if deadlock_counter >= self.deadlock_safe:
                    # then we are in a corner and barricaded ourselves
                    break

            last_pos = tmp_pos
            grid[
                last_pos[0],
                last_pos[1],
                last_pos[2]
            ] = 1

        return grid

    def grid_to_cubes(self, grid):
        vertices = []
        faces = []

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for z in range(self.grid_size):
                    if grid[x, y, z] == 1:
                        vert_, fac_ = self.make_cube()
                        vert_, fac_ = self.move_cube(vert_, x, y, z, fac_, len(vertices))
                        vertices += vert_
                        faces += fac_

        return vertices, faces

    def write_obj(self, grid, vertices, faces, out_path):
        if not os.path.exists(os.path.dirname(out_path)):
            try:
                os.makedirs(os.path.dirname(out_path))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

        with open(out_path, "w") as file_:
            file_.write("# made with _flo_\n\n")

            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    file_.write("# {}\n".format(" ".join([str(x) for x in grid[x, y]])))
                file_.write("# " + "==" * self.grid_size + "\n")

            file_.write("\n")

            for v in vertices:
                file_.write("v {} {} {}\n".format(*[float(x) for x in v]))

            file_.write("\n")

            for f in faces:
                file_.write("f {} {} {}\n".format(*[int(x + 1) for x in f]))

    def generate_obj(self, grid, vertices, faces):
        """Generates object string
        Returns: string representing the object for in-memory operation
        """
        file_ = io.StringIO()

        file_.write("# made with _flo_\n\n")

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                file_.write("# {}\n".format(" ".join([str(x) for x in grid[x, y]])))
            file_.write("# " + "==" * self.grid_size + "\n")

        file_.write("\n")

        for v in vertices:
            file_.write("v {} {} {}\n".format(*[float(x) for x in v]))

        file_.write("\n")

        for f in faces:
            file_.write("f {} {} {}\n".format(*[int(x + 1) for x in f]))

        return file_.getvalue()

    @staticmethod
    def center_grid(grid):
        nonzeros = np.array(np.nonzero(grid))
        minima = np.amin(nonzeros, axis=(1), keepdims=True).flatten()

        if np.count_nonzero(minima) != 0:
            for axis in range(3):
                grid = np.roll(grid, -minima[axis], axis)

        return grid
