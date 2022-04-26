# -*- coding: utf-8 -*-
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


def drawFunc():
    # Clear previous screen
    glClear(GL_COLOR_BUFFER_BIT)
    glRotatef(0.1, 5, 5, 0) # (angle,x,y,z)
    glutSolidTeapot(0.5) # solid teapot
    # Refresh display
    glFlush()


# Use glut to initialize OpenGL
glutInit()
# Display mode: GLUT_SINGLE unbuffered direct display|GLUT_RGBA uses RGB (A is not alpha)
glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA)
# Window position and size-generation
glutInitWindowPosition(0, 0)
glutInitWindowSize(400, 400)
glutCreateWindow(b"first")
# Call function to draw image
glutDisplayFunc(drawFunc)
glutIdleFunc(drawFunc)
# Main loop
glutMainLoop()