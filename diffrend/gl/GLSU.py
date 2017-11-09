# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 12:51:59 2015

@author: Fahim Mannan (fmannan@gmail.com)

GL Shader Utility
-----------------

usage:
shader_program = ShaderProgram()
shader_program.addShaderFromFile(GL_VERTEX_SHADER, vert_filename)
shader_program.addShaderFromFile(GL_FRAGMENT_SHADER, frag_filename)
shader_program.link()
shader_program.use()
shader_program.setValue('uniform', 'mvpMatrix', mvpMatrix)
shader_program.setValue('uniform', 'lightPos', lightPos)
shader_program.setValue('attribute', 'normal', normal) # per vertex attribute
"""
import OpenGL.GL as gl
import numpy as np

class ShaderObject:
    def __init__(self, shader_type):
        if not bool(gl.glCreateShader):
            print('unable to initialize')
        print(shader_type)
        self.type = shader_type
        self.id = gl.glCreateShader(self.type)
                
    def createFromSource(self, src):
        gl.glShaderSource(self.id, src)
        gl.glCompileShader(self.id)
        log = gl.glGetShaderInfoLog(self.id)
        if log: print('ShaderObject.createFromSource: ', log)
        return self
        
    def createFromFile(self, filename):
        with open(filename) as shader_file:
            shader_src = shader_file.read()
            print(shader_src)
            self.createFromSource(shader_src)
            return self
        
class ShaderProgram:
    def __init__(self):
        self.id = gl.glCreateProgram()
        log = gl.glGetProgramInfoLog(self.id)
        if log: print('ShaderProgram.__init__: ', log)
        self.ShaderObjects = {gl.GL_VERTEX_SHADER : None,
                              gl.GL_FRAGMENT_SHADER : None}
        self.uniform_variable_location_cache = {}
        self.attribute_variable_location_cache = {}
        
    def addShaderFromFile(self, shader_type, filename):
        #print('adding ' + filename)
        self.addShaderObject(ShaderObject(shader_type).createFromFile(filename) )
        
    def addShaderFromSource(self, shader_type, src):
        self.addShaderObject(ShaderObject(shader_type).createFromSource(src) )
        
        
    def addShaderObject(self, obj):
        #dprint(obj)
        self.ShaderObjects[obj.type] = obj
        gl.glAttachShader(self.id, obj.id)
#        log = gl.glGetShaderInfoLog(self.id)
#        if log: print 'ShaderProgram.addShaderObject: ', log
     
    def removeShaderObject(self, obj):
        gl.glDetachShader(self.id, obj.id)
        self.ShaderObjects[obj.type] = None
        
    def link(self):
        gl.glLinkProgram(self.id)
        status = gl.glGetProgramiv(self.id, gl.GL_LINK_STATUS)
        if(status):
            print('ShaderProgram linking successful.')
        else:
            print('ShaderProgram linking failed  ...')
        
        log = gl.glGetProgramInfoLog(self.id)
        if log: print('ShaderProgram.addShaderObject: ', log)
            
        gl.glValidateProgram(self.id)
        log = gl.glGetProgramInfoLog(self.id)
        if log: print('ShaderProgram.addShaderObject: ', log)

        # shader variables may change...so empty cache
        self.uniform_variable_location_cache = {}
        self.attribute_variable_location_cache = {} 
     
    def use(self):
        gl.glUseProgram(self.id)
    
    def setValueByLocation(self, var_location, var):
        # figure out which glUniform* function to call
        if type(var) is float:
            gl.glUniform1f(var_location, var)
        elif type(var) is int:
            gl.glUniform1i(var_location, var)
        elif type(var) is np.ndarray:
            if var.shape == (4, 4):
                gl.glUniformMatrix4fv(var_location, 1, gl.GL_FALSE, var)
            elif var.shape <= (4,):
                gl.glUniform4fv(var_location, 1, var)
        return var_location
                
    def setValue(self, var_type, var_name, var):
        res = None
        if var_type == 'uniform':
            res = self.setUniformValue(var_name, var)
        elif var_type == 'attribute':
            res = self.setAttributeValue(var_name, var)
        return res
        
    def setAttributeValue(self, var_name, var):
        if var_name not in self.attribute_variable_location_cache:
            self.attribute_variable_location_cache[var_name] = gl.glGetAttribLocation(self.id, var_name)
        var_location = self.attribute_variable_location_cache[var_name]
        if var_location < 0:
            print('Failed to find attribute variable : ' + var_name)
            return var_location
        return self.setValueByLocation(var_location, var)
    
    def setUniformValue(self, var_name, var):
        if var_name not in self.uniform_variable_location_cache:
            self.uniform_variable_location_cache[var_name] = gl.glGetUniformLocation(self.id, var_name)
            print('var_name', self.uniform_variable_location_cache[var_name])
        var_location = self.uniform_variable_location_cache[var_name]
        if var_location < 0:
            print('Failed to find uniform variable : ' + var_name)
            return var_location
        return self.setValueByLocation(var_location, var)


if __name__ == '__main__':
    from OpenGL.GLUT import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
    import sys
    
    quadric = None
    
    def display():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        
        glRotatef(45, 0., 0. , 1.)
        glScalef(0.5, 0.5, 0.5)
        
        glBegin(GL_TRIANGLES)
        glColor3f(0., 1., 1.)
        glVertex3f(-1, -1, 0.)
        glColor3f(1., 0., 1.)
        glVertex3f(1., -1., 0.)
        glColor3f(1., 1., 1.)
        glVertex3f(0, 1, 0.)
        glEnd()
        glPopMatrix()
        
        glPushMatrix()
        glTranslatef(0, .25, 0)
        glutSolidTeapot(0.25)
        glPopMatrix()
        
        glPushMatrix()
        glColor3f(1., 0., 0.)
        glTranslatef(-.5, -.5, -1)
        gluSphere(quadric, 0.5, 32, 32)
        glPopMatrix()
        
        glutSwapBuffers()
    
    def reshape(width,height):
        glViewport(0, 0, width, height)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        #glMatrixMode(GL_PROJECTION)
        #gluPerspective()
    
    def keyboard(key, x, y):
        if key == '\033':
            sys.exit()
    
    def init():
        glClearColor(0.0, 0.0, 0.0, 1.0)
        global quadric
        quadric = gluNewQuadric()
        print(quadric)
        
        glEnable(GL_DEPTH_TEST)
        
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)
    glutCreateWindow('GLSU (GL Shader Utility) Test')
    glutReshapeWindow(512, 512)
    
    glutReshapeFunc(reshape)
    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard)
    init()
    
    # setup shader
    # create a shader program
    # add vertex and fragment shaders to the program
    # link
    # use program
    # set variables
    # 
    shader_obj = ShaderObject(gl.GL_VERTEX_SHADER)
    shader_obj.createFromFile('./shaders/vertexShader.vert')
    
    shader_program = ShaderProgram()
    print('adding vertex shader ...')
    shader_program.addShaderObject(shader_obj)
    print('adding fragment shader ...')
    shader_program.addShaderFromFile(gl.GL_FRAGMENT_SHADER, './shaders/fragmentShader.frag')
    shader_program.link()
    print(gl.glValidateProgram(shader_program.id))
    shader_program.use()
    shader_program.setUniformValue('scale', 20.0)
    shader_program.setUniformValue('color', np.array([.25, 0, 0, 0]))
    print(glGetUniformLocation(shader_program.id, "scale"))
    print(glGetUniformLocation(shader_program.id, "color"))
    print(glGetAttribLocation(shader_program.id, 'vertex'))
    print(glGetAttribLocation(shader_program.id, 'scale'))
    print(glGetUniformLocation(shader_program.id, "mvpMatrix"))
    print(glGetUniformLocation(shader_program.id, "color"))
    #shader_program.setUniformValue("mvpMatrix", np.eye(4))
    
    glutMainLoop()    
