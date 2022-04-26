# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 14:51:41 2014
@author: ray
"""
from PIL import Image

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame, pygame.image
from pygame.locals import *
import numpy
import pickle
from q3flag_finalized import find_homography, apply_homography
import numpy as np
import cv2
from matplotlib import pyplot as plt
import camera
from objloader import OBJ

class ImageLoader:

  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.width = 0
    self.height = 0
    self.img_data = 0

  def load(self, image):
    im = image
    tx_image = cv2.flip(im, 0)
    tx_image = Image.fromarray(tx_image)
    self.width = tx_image.size[0]
    print(self.width)
    self.height = tx_image.size[1]
    print(self.height)
    
    self.img_data = tx_image.tobytes('raw', 'BGRX', 0, -1)

    self.Texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, self.Texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.img_data)

  def draw(self):
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslate(self.x, self.y, 0)
    glEnable(GL_TEXTURE_2D)  
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex2f(0, 0)
    glTexCoord2f(1, 0)
    glVertex2f(self.width, 0)
    glTexCoord2f(1, 1)
    glVertex2f(self.width, self.height)
    glTexCoord2f(0, 1)
    glVertex2f(0, self.height)
    glEnd()
    glDisable(GL_TEXTURE_2D)
    
    
def my_calibration(sz):
  row,col = sz
  fx = 1430*col/2592
  fy = 1430*row/1936
  k = np.diag([fx,fy,1])
  k[0,2] = 0.5*col
  k[1,2] = 0.5*row
  return k

def loadImgs_plus(path_im, keyword="", grayscale=False):
  fs = []
  fullfs = []

  files = os.listdir(path_im)
  print(path_im)
  files.sort(key=lambda f: int(re.sub('\D', '', f)))
  for file in files:
    if file.find(keyword) != -1:
      fs.append(file)
      fullfs.append(path_im + "/" + file)

  frame = len(fs)
  byte = 1
  if grayscale:
    im = cv2.imread(fullfs[0], 0)
    row, column = im.shape
  else:
    im = cv2.imread(fullfs[0])
    row, column, byte = im.shape

  imgs = np.zeros([frame, row, column, byte], dtype=np.float32).squeeze()

  for i in range(len(fullfs)):
    print("loading file:", fs[i], end='\r')
    if grayscale:
      im = cv2.imread(fullfs[i], 0)
    else:
      im = cv2.imread(fullfs[i])
    imgs[i] = im
  print("")

  return imgs


def set_projection_from_camera(K):
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()

  fx = float(K[0, 0])
  fy = float(K[1, 1])
  fovy = 2 * numpy.arctan(0.5 * height / fy) * 180 / numpy.pi
  aspect = (width * fy) / (height * fx)

  near, far = 0.1, 100
  gluPerspective(fovy, aspect, near, far)
  glViewport(0, 0, int(width), int(height))

def initTracker(img, corners):
  # initialize your tracker with the first frame from the sequence and
  # the corresponding corners from the ground truth
  # this function does not return anything
  global old_frame
  global p0
  old_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  p0 = corners.T.astype(np.float32)
  pass


def updateTracker(img):
  # update your tracker with the current image and return the current corners
  # at present it simply returns the actual corners with an offset so that
  # a valid value is returned for the code to run without errors
  # this is only for demonstration purpose and your code must NOT use actual corners in any way
  global old_frame
  global p0
  frame_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # parameters for lucas kanade optical flow

  lk_params = dict(winSize=(40, 40),
                     maxLevel=7,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 9, 0.03))    
  """
    lk_params = dict(winSize=(32, 32),
                     maxLevel=8,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 9, 0.03))
    """

  """
    _, pyr_old = cv2.buildOpticalFlowPyramid(old_frame, winSize=(15, 15), maxLevel=4)
    _, pyr_new = cv2.buildOpticalFlowPyramid(frame_img, winSize=(15, 15), maxLevel=4)    
    p1, st, err = cv2.calcOpticalFlowPyrLK(pyr_old, pyr_new, p0, None, **lk_params)
    """
  p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame_img, p0, None, **lk_params)
  old_frame = frame_img.copy()
  p0 = p1.copy()

  print(p1.T)
  return p1.T

def set_modelview_from_camera(Rt):
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()

  # Rotate 90 deg around x, so that z is up.
  Rx = numpy.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

  # Remove noise from rotation, make sure it's a pure rotation.
  R = Rt[:, :3]
  U, S, V = numpy.linalg.svd(R)
  R = numpy.dot(U, V)
  R[0, :] = -R[0, :]  # Change sign of x axis.

  print(S)
  t = Rt[:, 3]

  M = numpy.eye(4)
  M[:3, :3] = numpy.dot(R, Rx)
  M[:3, 3] = t

  m = M.T.flatten()
  glLoadMatrixf(m)


def draw_background(imname):
  bg_image = pygame.image.load(imname).convert()
  width, height = bg_image.get_size()
  bg_data = pygame.image.tostring(bg_image, "RGB", 1)

  glEnable(GL_TEXTURE_2D)
  tex = glGenTextures(1)
  glBindTexture(GL_TEXTURE_2D, tex)
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
               GL_RGBA, GL_UNSIGNED_BYTE, bg_data)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

  glBegin(GL_QUADS)
  glTexCoord2f(0, 0); glVertex3f(-1, -1, -1)
  glTexCoord2f(1, 0); glVertex3f( 1, -1, -1)
  glTexCoord2f(1, 1); glVertex3f( 1,  1, -1)
  glTexCoord2f(0, 1); glVertex3f(-1,  1, -1)
  glEnd()

  #texarray = (GLuint*1)(self.Texture_ID)
  #glDeleteTextures(1, texarray)
  #glDeleteTextures(tex)


def load_and_draw_model(filename):
  glEnable(GL_LIGHTING)
  glEnable(GL_LIGHT0)
  glEnable(GL_DEPTH_TEST)
  glClear(GL_DEPTH_BUFFER_BIT)
  glMaterialfv(GL_FRONT, GL_AMBIENT, [0, 0, 0, 0])
  glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0.75, 1, 0])
  glMaterialfv(GL_FRONT, GL_SHININESS, 0.25 * 128)
  import objloader
  obj = objloader.OBJ(filename, swapyz=True)
  glScale(0.000001, 0.000001, 0.000001)
  glCallList(obj.gl_list)


def setup():
  pygame.init()
  glutInit()
  pygame.display.set_mode((int(width), int(height)), OPENGL | DOUBLEBUF)
  pygame.display.set_caption('Look, an OpenGL window!')



width, height = 1240 / 2, 960 / 2
sequences = ['input_scene']
seq_id = 0

write_stats_to_file = 0
show_tracking_output = 1

seq_name = sequences[seq_id]
path = seq_name + '/%05d.jpg' % 0




img_base_frame = cv2.imread(str(path))
NrPoints = 4
cover_path = 'pure_bookCover.jpg'
book_cover = cv2.imread(str(cover_path))
plt.imshow(book_cover)

ref_cover = np.array([[0,0],[book_cover.shape[1] - 1, 0],[book_cover.shape[1] - 1, book_cover.shape[0] - 1],[0, book_cover.shape[0] - 1]])
cover = np.ones((3, NrPoints))

for i in range(NrPoints):
  cover[:2, i] = ref_cover[i]
  
  
plt.imshow(img_base_frame)

ref = plt.ginput(NrPoints)
print(ref)
init_corners = [list(ref[0]),
                  list(ref[1]),
                  list(ref[2]),
                  list(ref[3])]
init_corners = np.array(init_corners).T

X = np.ones((3, NrPoints))
for i in range(NrPoints):
  X[:2, i] = ref[i]
  

cover_homography = find_homography(cover, X, NrPoints, norm='euclidean',normalization=False)
initTracker(img_base_frame, init_corners)

image_base_frame_copy = np.copy(img_base_frame)
"""start pygame"""

K = my_calibration((int(book_cover.shape[0]), int(book_cover.shape[1])))
cam1 = camera.Camera(np.hstack((K,np.dot(K,np.array([[0],[0],[-1]])) )))

cam2 = camera.Camera(np.dot(cover_homography,cam1.P))
Rt = np.dot(np.linalg.inv(K),cam2.P)

setup()
print(seq_name + '/%05d.jpg' % 0)

draw_background(seq_name + '/%05d.jpg' % 0)

# FIXME: The origin ends up in a different place than in ch04_markerpose.py
# somehow.
set_projection_from_camera(K)
set_modelview_from_camera(Rt)

glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glEnable(GL_DEPTH_TEST)
glClear(GL_DEPTH_BUFFER_BIT)
glMaterialfv(GL_FRONT, GL_AMBIENT, [0, 0, 0, 0])
glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0, 0, 0])
glMaterialfv(GL_FRONT, GL_SHININESS, 0.25 * 128)
for y in range(0, 1):
  for x in range(0, 1):
    glutSolidTeapot(0.08)
    glTranslatef(0.04, 0, 0)
  glTranslatef(-3 * 0.04, 0, 0.04)
load_and_draw_model('toyplane.obj')

im_loader = ImageLoader(0, 0)


glClearColor(0.7, 0, 0, 1)
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluOrtho2D(0, width, height, 0)
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()



glDisable(GL_DEPTH_TEST)
im_loader.load(img_base_frame)
#glColor3f(1, 1, 1)
im_loader.draw()


pygame.display.flip()


while True:
  event = pygame.event.poll()
  if event.type in (QUIT, KEYDOWN):
    break
  pygame.display.flip()


#if __name__ == '__main__':
    #cap = cv2.VideoCapture(0)
    #width, height = int(cap.get(3)), int(cap.get(4))
    #pygame.init()
    #pygame.display.set_mode((width, height), pygame.DOUBLEBUF | pygame.OPENGL)
    ##box = OBJ('toyplane.obj',) # from OBJFileLoader import OBJ
    #im_loader = ImageLoader(0, 0)
    #angle = 0

    #glClearColor(0.7, 0, 0, 1)
    #run = True
    #while run:
        #for event in pygame.event.get():
            #if event.type == pygame.QUIT:
               #run = False

        #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        #glMatrixMode(GL_PROJECTION)
        #glLoadIdentity()
        #gluOrtho2D(0, width, height, 0)
        #glMatrixMode(GL_MODELVIEW)
        #glLoadIdentity()

        #glDisable(GL_DEPTH_TEST)
        #success, image = cap.read()
        #if success:
            #im_loader.load(image)
        #glColor3f(1, 1, 1)
        #im_loader.draw()
        
        #glMatrixMode(GL_PROJECTION)
        #glLoadIdentity()
        #gluPerspective(45, (width / height), 0.1, 50.0)
        #glMatrixMode(GL_MODELVIEW)
        #glLoadIdentity()
        #glTranslate(0.0, 0.0, -5)
        #glRotate(angle, 0, 1, 0)
        #angle += 1
        
        #glEnable(GL_DEPTH_TEST)
        ##box.render()
        #load_and_draw_model('toyplane.obj')
        #pygame.display.flip()

    #pygame.quit()
    #quit()