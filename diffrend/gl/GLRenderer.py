from PyQt5.QtCore import Qt
import PyQt5.QtGui as QtGui
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtOpenGL import QGLWidget
from PyQt5.QtGui import QOpenGLVersionProfile, QSurfaceFormat

import numpy as np
from diffrend.numpy.camera import TrackBallCamera
from diffrend.numpy.quaternion import Quaternion
import diffrend.numpy.ops as ops

from array import array
from OpenGL.GL import *
from OpenGL.GLU import *
import textwrap
import time
import os

from diffrend.gl import shaders
from diffrend.gl.GLSU import ShaderProgram


def get_val(dict_var, key, default_val):
    if key in dict_var:
        return dict_var[key]
    return default_val


def glLookat(eye, at, up):
    gluLookAt(eye[0], eye[1], eye[2], at[0], at[1], at[2], up[0], up[1], up[2])

def draw_triangle(vertices):
    pass

def draw_obj(obj):
    glBegin(GL_TRIANGLES)
    # render each face
    for face_idx in range(obj['f'].shape[0]):
        # gl.glColor3f(0.25, 0.25, 0.25)
        # gl.glNormal3f(-self.face_normals[face_idx][0], -self.face_normals[face_idx][1],
        #               -self.face_normals[face_idx][2])
        v = obj['v'][obj['f'][face_idx]]

        if 'fn' in obj:
            normal = obj['fn'][face_idx]
        else:
            v01 = v[1] - v[0]
            v02 = v[2] - v[0]
            # print(v)
            normal = ops.normalize(np.cross(v01, v02))

        for idx in range(3):
            glNormal3f(normal[0], normal[1], normal[2])
            glVertex3f(v[idx][0], v[idx][1], v[idx][2])

    glEnd()


def draw_disk(radius):
    quadric = gluNewQuadric()
    gluDisk(quadric, 0.0, radius, 32, 18)
    gluDeleteQuadric(quadric)

def draw_quad_xy():
    glBegin(GL_QUADS)
    glNormal3f(0.0, 0.0, 1.0)
    glVertex3f(-0.5, -0.5, 0.0)
    glVertex3f(0.5, -0.5, 0.0)
    glVertex3f(0.5, 0.5, 0.0)
    glVertex3f(-0.5, 0.5, 0.0)
    glEnd()


def draw_cube(scale):
    glPushMatrix()
    glScalef(scale, scale, scale)

    # glPushMatrix()
    # glTranslatef(0., 0.5, 0.)
    # glRotatef(-90., 1.0, 0.0, 0.0)
    # draw_quad_xy()
    # glPopMatrix()
    #
    # glPushMatrix()
    # glTranslatef(0., -0.5, 0.)
    # glRotatef(90., 1.0, 0.0, 0.0)
    #
    # draw_quad_xy()
    # glPopMatrix()

    glPushMatrix()
    glTranslatef(0., 0., 0.5)
    draw_quad_xy()
    glPopMatrix()
    #
    # glPushMatrix()
    # glRotatef(180., 0.0, 1.0, 0.0)
    # glTranslatef(0., 0.0, 0.5)
    # draw_quad_xy()
    # glPopMatrix()

    glPopMatrix()


def draw_sphere(radius):
    quadric = gluNewQuadric()
    gluSphere(quadric, radius, 36, 32)
    gluDeleteQuadric(quadric)


def draw_cylinder(scale, base=.25, top=.25, height=1.0, slices=36, stacks=18):
    glPushMatrix()
    glScalef(scale, scale, scale)
    quadric = gluNewQuadric()
    gluCylinder(quadric, base, top, height, slices, stacks)
    gluDeleteQuadric(quadric)
    glPopMatrix()


def draw_axes(height=1.):
    # z-axis (blue)
    glPushAttrib(GL_ALL_ATTRIB_BITS)
    glColor3f(0.0, 0.0, 1.0)
    draw_cylinder(.05, 0.1, 0.1, height)
    glPopAttrib()

    # x-axis (red)
    glPushAttrib(GL_ALL_ATTRIB_BITS)
    glColor3f(1.0, 0.0, 0.0)
    glPushMatrix()
    glRotatef(90, 0, 1, 0)
    draw_cylinder(.05, 0.1, 0.1, height)
    glPopMatrix()
    glPopAttrib()

    # y-axis (green)
    glPushAttrib(GL_ALL_ATTRIB_BITS)
    glColor3f(0.0, 1.0, 0.0)
    glPushMatrix()
    glRotatef(90, 1, 0, 0)
    draw_cylinder(.05, 0.1, 0.1, height)
    glPopMatrix()
    glPopAttrib()



class GL21Widget(QOpenGLWidget):
    def __init__(self, version_profile=None, parent=None, **kwargs):
        super(GL21Widget, self).__init__(parent)
        self.version_profile = version_profile
        self.obj = kwargs['obj']
        self.obj['fn'] = ops.compute_face_normal(self.obj)

        self.scene = kwargs['scene'] if 'scene' in kwargs else None
        self.clear_color = get_val(kwargs, 'clear_color', [0.1, 0.1, 0.1, 1.0])
        self.model_view = np.eye(4)
        self.camera = TrackBallCamera([20, 20, 20, 1], at=[0, 0, 0, 1], up=[0, 1, 0, 0], fovy=np.deg2rad(45),
                                      focal_length=1., viewport=[0, 0, 640, 480])

    def initializeGL(self):
        self.gl = self.context().versionFunctions(self.version_profile)
        self.gl.initializeOpenGLFunctions()

        self.gl.glClearColor(self.clear_color[0], self.clear_color[1], self.clear_color[2], self.clear_color[3])
        self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        #glEnable(GL_CULL_FACE | GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)

        ambientLight = [0.2, 0.2, 0.2, 1.0]
        diffuseLight = [0.8, 0.8, 0.8, 1.0]
        specularLight = [0.5, 0.5, 0.5, 1.0]
        light_pos = [100.0, 0.0, -100.0, 1.0]

        glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight)
        #glLightfv(GL_LIGHT0, GL_SPECULAR, specularLight)
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)

        glEnable(GL_COLOR_MATERIAL)
        colorGray = [0.5, 0.5, 0.5, 1.0]
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, colorGray)

    def paintGL(self):
        gl = self.gl
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.camera.fovy, self.camera.viewport[2] / self.camera.viewport[3], 0.1, 1000.)
        matrix = glGetDoublev(GL_PROJECTION_MATRIX)
        print('projection matrix\n', matrix)
        print(self.camera.projection)

        glMatrixMode(GL_MODELVIEW)
        # mat = gl.glGetDoublev(gl.GL_PROJECTION_MATRIX)
        # print(type(mat), mat)

        # gl.glRotatef(45.0, 0.0, 0.0, 1.0)
        # gl.glScalef(1.5, 1.5, 1.5)
        glLoadIdentity()
        glLookat(self.camera.pos, self.camera.at, self.camera.up)
        glScalef(.5, .5, .5)

        # matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
        # print(type(matrix), matrix)
        # print(self.camera.model_view)
        # print(self.camera.M)
        # m = np.eye(4)
        # m[:3, 3] = np.array([0, 0, -1.])
        # glMultMatrixd(m.T)
        # matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
        #print(matrix)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)


        draw_sphere(.05)
        draw_cube(.1)
        #draw_cylinder(.05)
        draw_axes(2.)
        draw_disk(.05)

        if self.obj is None:
            gl.glBegin(gl.GL_TRIANGLES)
            #gl.glColor3f(0.5, 0.6, 0.3)
            gl.glNormal3f(0, 0, -1)
            gl.glVertex3f(0, 0, 0)
            gl.glVertex3f(1, 0, 0)
            gl.glVertex3f(0, 1, 0.5)
            gl.glEnd()
        else:
            pass
            #draw_obj(self.obj)


    def resizeGL(self, w, h):
        gl = self.gl
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glViewport(0, 0, w, h)
        self.camera.viewport = [0, 0, w, h]

    def rotate_model(self, step):
        pass

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() == Qt.Key_Escape:
            self.close()

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
        print(self.camera)
        self.repaint()

    def wheelEvent(self, e: QtGui.QWheelEvent):
        #print('wheel ', e.pixelDelta().x(), e.pixelDelta().y(), e.x(), e.y(), e.phase(), e.angleDelta())
        self.camera.zoom(-e.angleDelta().y() / 100.)
        print(self.camera)
        self.repaint()


class GLRenderer(QGLWidget):
    def __init__(self, version_profile=None, parent=None, **kwargs):
        super(GLRenderer, self).__init__(parent)
        # TODO: Instead of loading an object, load a scene file containing all the objects, shaders, lighting, etc.
        self.obj = kwargs['obj']
        self.scene = kwargs['scene'] if 'scene' in kwargs else None
        self.clear_color = get_val(kwargs, 'clear_color', [0.1, 0.1, 0.1, 1.0])
        self.model_view = np.eye(4)
        self.camera = TrackBallCamera([0, 10, 10, 1], at=[0, 0, 0, 1], up=[0, 1, 0, 0], fovy=np.deg2rad(45),
                                      focal_length=1., viewport=[0, 0, 640, 480])

        self.vs_src = os.path.join(shaders.DIR_SHADERS, 'vs.vert')  #kwargs['vs_src']
        self.fs_src = os.path.join(shaders.DIR_SHADERS, 'fs.frag')  #kwargs['fs_src']

        self.start_time = time.clock()
        self.startTimer(0)

    @staticmethod
    def _opengl_info():
        return textwrap.dedent("""\
            Vendor: {0}
            Renderer: {1}
            OpenGL Version: {2}
            Shader Version: {3}
            Num Extensions: {4}
            Extensions: {5}
        """).format(
            glGetString(GL_VENDOR).decode("utf-8"),
            glGetString(GL_RENDERER).decode("utf-8"),
            glGetString(GL_VERSION).decode("utf-8"),
            glGetString(GL_SHADING_LANGUAGE_VERSION).decode("utf-8"),
            glGetIntegerv(GL_NUM_EXTENSIONS),
            glGetString(GL_EXTENSIONS)
        )

    def initializeGL(self):
        print('{:*^80}'.format('Opengl Context ready'))
        print(self._opengl_info())

        glClearColor(self.clear_color[0], self.clear_color[1], self.clear_color[2], self.clear_color[3])
        glClear(GL_COLOR_BUFFER_BIT)

        shader_program = ShaderProgram()
        shader_program.addShaderFromFile(GL_VERTEX_SHADER, self.vs_src)
        shader_program.addShaderFromFile(GL_FRAGMENT_SHADER, self.fs_src)
        shader_program.link()
        self.shader_program = shader_program
        #shader_program.use()
        # shader_program.setValue('uniform', 'mvpMatrix', mvpMatrix)
        # shader_program.setValue('uniform', 'lightPos', lightPos)
        # shader_program.setValue('attribute', 'normal', normal)  # per vertex attribute

        vertices = [
            0.0, 0.5, 0.0,
            0.5, -0.5, 0.0,
            -0.5, -0.5, 0.0
        ]

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        position = glGetAttribLocation(self.shader_program.id, 'position')

        glEnableVertexAttribArray(position)
        glVertexAttribPointer(
            position, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
        glBufferData(
            GL_ARRAY_BUFFER, array("f", vertices).tobytes(), GL_STATIC_DRAW)

        glBindVertexArray(0)
        glDisableVertexAttribArray(position)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def timerEvent(self, event):
        elapsed = time.clock() - self.start_time
        self.repaint()

    def resizeGL(self, w: int, h: int):
        self.camera.viewport = [0, 0, w, h]

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        #glUseProgram(self.program)
        self.shader_program.use()

        # set the updated transformation matrices
        mvMatrix = np.eye(4)  # self.camera.model_view
        self.shader_program.setUniformValue('model_view', mvMatrix)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)

        glUseProgram(0)

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() == Qt.Key_Escape:
            self.close()

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
        print(self.camera)
        self.repaint()
        # project to 3d sphere
        # ..

    def wheelEvent(self, e: QtGui.QWheelEvent):
        # print('wheel ', e.pixelDelta().x(), e.pixelDelta().y(), e.x(), e.y(), e.phase(), e.angleDelta())
        self.camera.zoom(-e.angleDelta().y() / 1000.)
        print(self.camera)
        self.repaint()


def create_OGL_window(major, minor, obj, samples=4, clear_color=[0.0, 0.0, 0.0, 1.0], title=''):
    if major == None and minor == None:
        #window = GLDefaultWidget(obj=obj)
        window = GLRenderer(obj=obj)
    else:
        fmt = QSurfaceFormat()
        fmt.setVersion(major, minor)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setSamples(samples)
        QSurfaceFormat.setDefaultFormat(fmt)

        vp = QOpenGLVersionProfile()
        vp.setVersion(major, minor)
        vp.setProfile(QSurfaceFormat.CoreProfile)

        window = GL21Widget(version_profile=vp, clear_color=clear_color, obj=obj)
    window.setWindowTitle(title)
    return window


if __name__ == '__main__':
    import sys
    from PyQt5.Qt import QApplication
    from diffrend.model import load_obj
    
    app = QApplication(sys.argv)

    obj = load_obj('../../data/bunny.obj')
    window = create_OGL_window(2, 1, obj=obj, title='OpenGL 2.1 Renderer')

    # if args.renderer == 'gl21':
    #     window = create_OGL_window(2, 1, obj=obj, title='OpenGL 2.1 Renderer')
    # elif args.renderer == 'gl':
    #     window = create_OGL_window(None, None, obj=obj, title='OpenGL Renderer')
    # else:
    #     raise ValueError('Renderer {} not implemented.'.format(args.rederer))

    window.show()

    sys.exit(app.exec())