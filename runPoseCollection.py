import socket
from Measurement import Measurement
import numpy as np
import math as mathf
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *


def init():
    # Initialize openGL Graphics
    glShadeModel(GL_SMOOTH)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)


def drawText(position, textString):
    font = pygame.font.SysFont("Courier", 18, True)
    textSurface = font.render(textString, True, (255, 255, 255, 255), (0, 0, 0, 255))
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glRasterPos3d(*position)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)


def draw():
    global rquad
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glLoadIdentity()
    glTranslatef(0, 0.0, -7.0)

    osd_text = "pitch: " + str("{0:.2f}".format(ay)) + ", roll: " + str("{0:.2f}".format(ax))

    if yaw_mode:
        osd_line = osd_text + ", yaw: " + str("{0:.2f}".format(az))
    else:
        osd_line = osd_text

    drawText((-2, -2, 2), osd_line)

    rotMat1 = np.array([rotMat[0,:],rotMat[2,:],rotMat[1,:],rotMat[:,3]])
    glMultMatrixf(rotMat1)

    glBegin(GL_QUADS)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(1.0, 0.2, -1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(1.0, 0.2, 1.0)

    glColor3f(1.0, 0.5, 0.0)
    glVertex3f(1.0, -0.2, 1.0)
    glVertex3f(-1.0, -0.2, 1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(1.0, -0.2, -1.0)

    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(1.0, 0.2, 1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(-1.0, -0.2, 1.0)
    glVertex3f(1.0, -0.2, 1.0)

    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(1.0, -0.2, -1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(1.0, 0.2, -1.0)

    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(-1.0, -0.2, 1.0)

    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(1.0, 0.2, -1.0)
    glVertex3f(1.0, 0.2, 1.0)
    glVertex3f(1.0, -0.2, 1.0)
    glVertex3f(1.0, -0.2, -1.0)
    glEnd()

def resize(width, height):
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0 * width / height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def establishConn(hostID):

    try:
        host = hostID
        port = 5555

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.bind((host, port))
    except():
        print('Please connect to the IPv4 address')


    return s

hostIP = '192.168.42.2'
s = establishConn(hostIP)

measurementObj = Measurement(s)
# Initialize Arrays
# mag = np.array([0, 0, 0], dtype=float)
# mag.shape = (1, 3)
# linearAccel = np.array([0, 0, 0], dtype=float)
# linearAccel.shape = (1, 3)
# rotRate = np.array([0, 0, 0, 0], dtype=float)
# rotRate.shape = (1, 4)

# wPrev = np.array([0, 0, 0]) # Vector containing previous time step rotation rate

# Initialize time
message, address = s.recvfrom(8192)
t1 = message.decode("utf-8").replace(',', '').split()[0]
timeValue1 = float(t1)
measurementObj.t1 = timeValue1
# Initialize state status
# initialized = 0
# do_bias_estimation = 1
# bias_alpha = 0.01

# Initialize bias in rotation rate measurements
# wx_bias = 0
# wy_bias = 0
# wz_bias = 0
#
# # Initialize Constants
# Gravity = 9.81
# wThreshhold = 0.2
# dwThreshhold = 0.01
# accThreshold = 0.1
# gamma = 0.01
# epsilon = 0.9

# Initialize pygame for viewing phone rotation
video_flags = OPENGL | DOUBLEBUF
pygame.init()
screen = pygame.display.set_mode((640, 480), video_flags)
pygame.display.set_caption("Press Esc to quit, z toggles yaw mode")
resize(640, 480)
init()
frames = 0
ticks = pygame.time.get_ticks()
ax = ay = az = 0.0
yaw_mode = True

while 1:
    measurementObj.updatePose()
    qInit = measurementObj.qInit
    eulerAngles = measurementObj.eulerAngles() * 180 / np.pi

    # Display in openGL
    event = pygame.event.poll()
    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
        pygame.quit()  # * quit pygame properly
        break
    if event.type == KEYDOWN and event.key == K_z:
        yaw_mode = not yaw_mode

    ax = eulerAngles[0]
    ay = eulerAngles[1]
    az = eulerAngles[2]-.13
    rotMat = measurementObj.ep2Rot(measurementObj.qInit)
    draw()

    pygame.display.flip()
    frames = frames + 1