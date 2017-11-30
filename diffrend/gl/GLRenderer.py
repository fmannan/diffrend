from PyQt5.QtCore import Qt
import PyQt5.QtGui as QtGui
from PyQt5.QtWidgets import QOpenGLWidget, QFileDialog
from PyQt5.QtGui import QOpenGLVersionProfile, QSurfaceFormat

import numpy as np
from diffrend.model import load_model, load_splat
from diffrend.numpy.camera import TrackBallCamera
import diffrend.numpy.ops as ops
from diffrend.utils.utils import get_param_value
from data import DIR_DATA
from array import array
from OpenGL.GL import *
import textwrap
import time
import os

from diffrend.gl import shaders
from diffrend.gl.GLSU import ShaderProgram


def generate_disk_mesh(radius, samples=36):
    vertices = []
    faces = []
    vertices.append([0, 0, 0])
    for idx in range(samples):
        theta = 2 * np.pi * idx / samples
        vertices.append(np.array([np.cos(theta), np.sin(theta), 0]) * radius)
        if idx > 0:
            faces.append(np.array([0, idx, idx + 1]))
    faces.append(np.array([0, samples, 1]))
    vertices = np.array(vertices)
    faces = np.array(faces)
    return vertices, faces


class Mesh:
    def __init__(self, filename, **kwargs):
        print('Mesh::__init__')
        self.pos_loc = get_param_value('position', kwargs, None, required=True)
        self.bbox = get_param_value('bbox', kwargs, False)
        self.vao = None
        self.vbo_pos = None
        self.ibo = None
        self.load_model(filename)

    def cleanup(self):
        if self.vao is not None:
            glDeleteVertexArrays(1, self.vao)
        if self.vbo_pos is not None:
            glDeleteBuffers(self.vbo_pos)
        if self.ibo is not None:
            glDeleteBuffers(self.ibo)

    def load_model(self, filename):
        print("Mesh::load_model Loading {}".format(filename))
        self.obj = load_model(filename)
        """Create VAO and VBOs
        """
        self.vertices = self.obj['v']

        if self.bbox is not None and self.bbox is not False:
            P = self.vertices
            P = (P - np.mean(P, axis=0)) / np.max(np.max(P, axis=0) - np.min(P, axis=0))
            self.vertices = P

        self.faces = self.obj['f']

        #self.cleanup()

        if self.vao is None:
            self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        if self.vbo_pos is None:
            self.vbo_pos = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        glEnableVertexAttribArray(self.pos_loc)
        glVertexAttribPointer(self.pos_loc, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
        glBufferData(GL_ARRAY_BUFFER, array("f", self.vertices.ravel()).tobytes(), GL_STATIC_DRAW)

        if self.ibo is None:
            self.ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, array("I", self.faces.ravel()).tobytes(), GL_STATIC_DRAW)
        glBindVertexArray(0)
        print('Mesh::load_model (LOADED)')

    def render(self, render_point_cloud=False):
        glBindVertexArray(self.vao)
        if not render_point_cloud:
            glDrawElements(GL_TRIANGLES, self.faces.size, GL_UNSIGNED_INT, ctypes.c_void_p(0))
            # glDrawArrays(GL_TRIANGLES, 0, self.vertices.size)
        else:
            glDrawArrays(GL_POINTS, 0, self.vertices.size)

        glBindVertexArray(0)


class Disk:
    def __init__(self, radius, samples=36, position=None):
        print('Disk::__init__')
        print('pos ', position)
        v, self.faces = generate_disk_mesh(radius=radius, samples=samples)
        v = np.concatenate((v, v[1][np.newaxis, :]), axis=0)
        self.vertices = v
        """Create VAO and VBOs
        """
        self.pos_loc = position
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo_pos = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        glEnableVertexAttribArray(self.pos_loc)
        glVertexAttribPointer(self.pos_loc, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
        glBufferData(GL_ARRAY_BUFFER, array("f", self.vertices.ravel()).tobytes(), GL_STATIC_DRAW)

    def render(self):
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_FAN, 0, self.vertices.shape[0])
        glBindVertexArray(0)


class Splat:
    # TODO: REPLACE THE MESH STUFF AND ADD INSTANCED DISK RENDERING WITH THE MODEL TRANSFORM STORED FOR EACH INSTANCE
    def __init__(self, filename, **kwargs):
        print('Splat::__init__')
        self.pos_loc = get_param_value('position', kwargs, None, required=True)
        self.bbox = get_param_value('bbox', kwargs, False)
        self.vao = None
        self.vbo_pos = None
        self.ibo = None
        self.load_model(filename)

    def cleanup(self):
        if self.vao is not None:
            glDeleteVertexArrays(1, self.vao)
        if self.vbo_pos is not None:
            glDeleteBuffers(self.vbo_pos)
        if self.ibo is not None:
            glDeleteBuffers(self.ibo)

    def load_model(self, filename):
        print("Splat::load_model Loading {}".format(filename))
        self.obj = load_splat(filename)
        """Create VAO and VBOs
        """
        self.vertices = self.obj['v']

        if self.bbox is not None and self.bbox is not False:
            P = self.vertices
            P = (P - np.mean(P, axis=0)) / np.max(np.max(P, axis=0) - np.min(P, axis=0))
            self.vertices = P

        self.faces = self.obj['f']

        # self.cleanup()

        if self.vao is None:
            self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        if self.vbo_pos is None:
            self.vbo_pos = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        glEnableVertexAttribArray(self.pos_loc)
        glVertexAttribPointer(self.pos_loc, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
        glBufferData(GL_ARRAY_BUFFER, array("f", self.vertices.ravel()).tobytes(), GL_STATIC_DRAW)

        if self.ibo is None:
            self.ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, array("I", self.faces.ravel()).tobytes(), GL_STATIC_DRAW)
        glBindVertexArray(0)
        print('Splat::load_model (LOADED)')

    def render(self, render_point_cloud=False):
        glBindVertexArray(self.vao)
        if not render_point_cloud:
            glDrawElements(GL_TRIANGLES, self.faces.size, GL_UNSIGNED_INT, ctypes.c_void_p(0))
            # glDrawArrays(GL_TRIANGLES, 0, self.vertices.size)
        else:
            glDrawArrays(GL_POINTS, 0, self.vertices.size)

        glBindVertexArray(0)


class GLRenderer(QOpenGLWidget):
    def __init__(self, version_profile=None, parent=None, **kwargs):
        super(GLRenderer, self).__init__(parent)
        # TODO: Instead of loading an object, load a scene file containing all the objects, shaders, lighting, etc.
        self.obj_filename = kwargs['obj_filename']
        self.mesh = None
        self.bbox = True  # rescale the object
        self.scene = kwargs['scene'] if 'scene' in kwargs else None
        self.clear_color = get_param_value('clear_color', kwargs, [0.1, 0.1, 0.1, 1.0])
        self.model_view = np.eye(4)
        self.camera = TrackBallCamera([0, 0, 2, 1], up=[0, 1, 0, 0], fovy=np.deg2rad(45),
                                      focal_length=0.01, viewport=[0, 0, self.width(), self.height()])
        self.render_point_cloud = False
        self.vs_src = os.path.join(shaders.DIR_SHADERS, 'vs.vert')  #kwargs['vs_src']
        self.fs_src = os.path.join(shaders.DIR_SHADERS, 'fs.frag')  #kwargs['fs_src']

        self.start_time = time.clock()
        self.startTimer(0)

    @staticmethod
    def _opengl_info():
        """
         Extensions: {5}
        glGetString(GL_EXTENSIONS)
        """
        return textwrap.dedent("""\
            Vendor: {0}
            Renderer: {1}
            OpenGL Version: {2}
            Shader Version: {3}
            Num Extensions: {4}
           
        """).format(
            glGetString(GL_VENDOR).decode("utf-8"),
            glGetString(GL_RENDERER).decode("utf-8"),
            glGetString(GL_VERSION).decode("utf-8"),
            glGetString(GL_SHADING_LANGUAGE_VERSION).decode("utf-8"),
            glGetIntegerv(GL_NUM_EXTENSIONS)
        )

    def initializeGL(self):
        print('{:*^80}'.format('Opengl Context ready'))
        print(self._opengl_info())

        glClearColor(self.clear_color[0], self.clear_color[1], self.clear_color[2], self.clear_color[3])

        glPointSize(2.0)
        glEnable(GL_DEPTH_TEST)
        #glEnable(GL_LESS)
        shader_program = ShaderProgram()
        shader_program.addShaderFromFile(GL_VERTEX_SHADER, self.vs_src)
        shader_program.addShaderFromFile(GL_FRAGMENT_SHADER, self.fs_src)
        shader_program.link()
        self.shader_program = shader_program

        pos_loc = glGetAttribLocation(self.shader_program.id, 'position')
        self.pos_loc = pos_loc
        # TODO: normal, lighting


        if self.obj_filename is not None:
            self.mesh = Mesh(self.obj_filename, position=pos_loc, bbox=self.bbox)

        self.disk = Disk(0.5, 36, position=pos_loc)
        # glBindVertexArray(0)
        # glDisableVertexAttribArray(position)
        # glBindBuffer(GL_ARRAY_BUFFER, 0)

    def timerEvent(self, event):
        elapsed = time.clock() - self.start_time
        self.repaint()

    def resizeGL(self, w: int, h: int):
        self.camera.viewport = [0, 0, w, h]
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        self.shader_program.use()

        # set the updated transformation matrices
        model = self.camera.model_matrix
        #print('model matrix', model)
        view = ops.lookat(eye=self.camera.pos, at=self.camera.at, up=self.camera.up)
        proj = ops.perspective(self.camera.fovy, self.camera.aspect_ratio, .01, 100.)

        self.shader_program.setUniformValue('model', model.T)
        self.shader_program.setUniformValue('view', view.T)
        self.shader_program.setUniformValue('projection', proj.T)
        #self.disk.render()
        self.mesh.render(self.render_point_cloud)

        glBindVertexArray(0)
        glUseProgram(0)

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() == Qt.Key_Escape:
            self.close()
        elif e.key() == Qt.Key_L:
            filename, _ = QFileDialog.getOpenFileName(self, "Load Model", DIR_DATA)
            if self.mesh is None:
                self.mesh = Mesh(filename=filename, position=self.pos_loc, bbox=self.bbox)
            self.mesh.load_model(filename)
            self.repaint()
        elif e.key() == Qt.Key_P:
            # toggle point cloud rendering
            self.render_point_cloud = not self.render_point_cloud

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        self.mouse_pos = [e.x(), e.y()]
        self.mouse_press = True
        self.camera.mouse_press(self.mouse_pos)

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent):
        self.mouse_press = False

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        x, y = e.x(), e.y()
        dx, dy = x - self.mouse_pos[0], y - self.mouse_pos[1]
        text = "x: {0}, y: {1}, dx: {2}, dy: {3}".format(x, y, dx, dy)
        self.setWindowTitle(text)
        self.camera.mouse_move([x, y])
        #print(self.camera)
        self.repaint()

    def wheelEvent(self, e: QtGui.QWheelEvent):
        # print('wheel ', e.pixelDelta().x(), e.pixelDelta().y(), e.x(), e.y(), e.phase(), e.angleDelta())
        self.camera.zoom(-e.angleDelta().y() / 1000.)
        #print(self.camera)
        self.repaint()


def create_OGL_window(major, minor, obj_filename, samples=4, clear_color=[0.0, 0.0, 0.0, 1.0], title=''):
    if major == None and minor == None:
        window = GLRenderer(obj_filename=obj_filename)
    else:
        from diffrend.gl.GL21Renderer import GL21Widget
        fmt = QSurfaceFormat()
        fmt.setVersion(major, minor)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setSamples(samples)
        QSurfaceFormat.setDefaultFormat(fmt)

        vp = QOpenGLVersionProfile()
        vp.setVersion(major, minor)
        vp.setProfile(QSurfaceFormat.CoreProfile)

        window = GL21Widget(version_profile=vp, clear_color=clear_color, obj_filename=obj_filename)
    window.setWindowTitle(title)
    return window


if __name__ == '__main__':
    import sys
    from PyQt5.Qt import QApplication

    app = QApplication(sys.argv)

    window = create_OGL_window(None, None, obj_filename=DIR_DATA + '/bunny.obj', title='Object Visualizer')

    # if args.renderer == 'gl21':
    #     window = create_OGL_window(2, 1, obj=obj, title='OpenGL 2.1 Renderer')
    # elif args.renderer == 'gl':
    #     window = create_OGL_window(None, None, obj=obj, title='OpenGL Renderer')
    # else:
    #     raise ValueError('Renderer {} not implemented.'.format(args.rederer))

    window.show()

    sys.exit(app.exec())

