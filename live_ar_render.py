"""
Todo1: user can choose ratio of the template/cover and set length ?

"""

from argparse import ArgumentParser
import os
import sys
import cv2
import numpy as np
import math
import time
import re
import random
from matplotlib import pyplot as plt
from random import randint
from q3flag_finalized import find_homography, apply_homography
from objloader_simple import *


DEFAULT_COLOR = (59, 59, 66)


def nothing(x):
    pass


def get_args_from_command_line():
    parser = ArgumentParser(description='params for AR')
    parser.add_argument('--obj',
                        dest='obj',
                        help='Path_of_obj_file_to_render_in_AR',
                        default='./result/test/model.obj',
                        type=str)

    parser.add_argument('--test_user_param_before_rendering',
                        dest='test_param',
                        help='if test the interactive parameters before live tracking and ar rendering,0 for no, 1 for yes',
                        default=0,
                        type=int)
    
    parser.add_argument('--intrinsic',
                        dest='intrinsic',
                        nargs='+',
                        help='intrinsic parameters of your camera',
                        default=[715,715,320,240],
                        type=int)
    
    parser.add_argument('--template_type',
                        dest='template_type',
                        help='template type,1 for square, 2 for rectangle with height greater than width, 3 for rectangle with width greater than height',
                        default=1,
                        type=int)    

    parser.add_argument('--camera_id',
                        dest='camera_id',
                        help='camera_id used for open cv camera capture',
                        default=0,
                        type=int)  
    
    
    parser.add_argument('--show_planar_tracking',
                        dest='show_planar_tracking',
                        help='if draw lines for tracked planar, 0 for no, 1 for yes, default 0',
                        default=0,
                        type=int)      
    
    args = parser.parse_args()
    return args

# named ites for easy reference
barsWindow = 'Bars'
ratio_bar = 'ratio(%)'
scale_bar = 'scale(%)'
x_bar = 'ratation_x'
y_bar = 'ratation_y'
z_bar = 'ratation_z'
t_1 = "t_x-50"
t_2 = "t_y-50"
t_3 = "t_z-50"
r = "blue"
g = "green"
b = "red"
pointDir_bar = 'reset_tracker'


# create window for the slidebars
cv2.namedWindow(barsWindow, flags = cv2.WINDOW_NORMAL)

# create the sliders
cv2.createTrackbar(ratio_bar, barsWindow, 0, 300, nothing)
cv2.createTrackbar(scale_bar, barsWindow, 0, 4000, nothing)
cv2.createTrackbar(x_bar, barsWindow, 0, 360, nothing)
cv2.createTrackbar(y_bar, barsWindow, 0, 360, nothing)
cv2.createTrackbar(z_bar, barsWindow, 0, 360, nothing)
cv2.createTrackbar(pointDir_bar, barsWindow, 0, 1, nothing)
cv2.createTrackbar(t_1, barsWindow, -100, 100, nothing)
cv2.createTrackbar(t_2, barsWindow, -100, 100, nothing)
cv2.createTrackbar(t_3, barsWindow, -100, 100, nothing)
cv2.createTrackbar(r, barsWindow, 0, 255, nothing)
cv2.createTrackbar(g, barsWindow, 0, 255, nothing)
cv2.createTrackbar(b, barsWindow, 0, 255, nothing)

# set initial values for sliders
cv2.setTrackbarPos(ratio_bar, barsWindow, 100)
cv2.setTrackbarPos(scale_bar, barsWindow, 100)
cv2.setTrackbarPos(x_bar, barsWindow, 0)
cv2.setTrackbarPos(y_bar, barsWindow, 0)
cv2.setTrackbarPos(z_bar, barsWindow, 0)
cv2.setTrackbarPos(pointDir_bar, barsWindow, 0)
cv2.setTrackbarPos(t_1, barsWindow, 50)
cv2.setTrackbarPos(t_2, barsWindow, 50)
cv2.setTrackbarPos(t_3, barsWindow, 50)
cv2.setTrackbarPos(r, barsWindow, 59)
cv2.setTrackbarPos(g, barsWindow, 59)
cv2.setTrackbarPos(b, barsWindow, 59)

args = get_args_from_command_line()


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


def drawRegion(img, corners, color, thickness=1):
    # draw the bounding box specified by the given corners
    for i in range(4):
        p1 = (int(corners[0, i]), int(corners[1, i]))
        p2 = (int(corners[0, (i + 1) % 4]), int(corners[1, (i + 1) % 4]))
        cv2.line(img, p1, p2, color, thickness)


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
    
    #lk_params = dict(winSize=(80, 80),
                     #maxLevel=2,
                     #criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 9, 0.025))    
    
    #lk_params = dict(winSize=(32, 32),
                     #maxLevel=8,
                     #criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 9, 0.03))
    #lk_params = dict(winSize=(100, 100),
                     #maxLevel=1,
                     #criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.028))
        
    lk_params = dict(winSize=(32, 32),
                     maxLevel=8,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 9, 0.03))    
    """
    _, pyr_old = cv2.buildOpticalFlowPyramid(old_frame, winSize=(15, 15), maxLevel=4)
    _, pyr_new = cv2.buildOpticalFlowPyramid(frame_img, winSize=(15, 15), maxLevel=4)    
    p1, st, err = cv2.calcOpticalFlowPyrLK(pyr_old, pyr_new, p0, None, **lk_params)
    """
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame_img, p0, None, **lk_params)
    old_frame = frame_img.copy()
    p0 = p1.copy()

    #print(p1.T)
    return p1.T


def main():
    """
    addded part for model projection preparation

    """
    #camera_parameters = np.array([[1430, 0, 480], [0, 1430, 640], [0, 0, 1]])
    print(args)
    cam_intrinsic = args.intrinsic
    
    camera_parameters = np.array([[715, 0, 320], [0, 715, 240], [0, 0, 1]])
    
    camera_parameters = np.array([[cam_intrinsic[0], 0, cam_intrinsic[2]], [0, cam_intrinsic[1], cam_intrinsic[3]], [0, 0, 1]])
    
    #camera_parameters = np.array([[715, 0, 480], [0, 715, 620], [0, 0, 1]])
    
    # Load 3D model from OBJ file
    
    obj_path = args.obj
    camera_id = args.camera_id
    obj = OBJ(obj_path, swapyz=True)    
    show_planar_tracking = args.show_planar_tracking
    
    ################## if apply ar to existing video

    """
    end of this part
    """


    NrPoints = 4

    #cover_path = 'BlackBackGround.jpg'
    # read the ground truth
    cover_path = './reference/ar_tag.png'
    template_type = args.template_type
    
    book_cover = cv2.imread(str(cover_path))
    plt.imshow(book_cover)
    
    temp_width = 300
    temp_height = 300
    if template_type == 1:
        pass
    elif template_type == 2:
        temp_height = 1.5 * temp_width
        
    elif template_type == 3:
        temp_width = 1.5 * temp_height
        
    
    
    ref_cover = np.array([[0,0],[temp_width, 0],[temp_width, temp_height],[0, temp_height]])
        
    #ref_cover = np.array([[0,0],[book_cover.shape[1] - 1, 0],[book_cover.shape[1] - 1, book_cover.shape[0] - 1],[0, book_cover.shape[0] - 1]])
    cover = np.ones((3, NrPoints))

    for i in range(NrPoints):
        cover[:2, i] = ref_cover[i]



    cap = cv2.VideoCapture(camera_id)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_FPS, 60)    
    cv2.namedWindow("capture frame")    
    while True:
        # read the current frame
        ret, frame = cap.read()
        if not ret:
            print("failed to capture the image")
            return 
        
        cv2.imshow("Press ESC to captured initial position for tracking", frame)
    
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ASCII:ESC pressed
            ret = ret
            img_base_frame = frame
            init_img = frame
            print("Escape hit, closing...")
            cv2.destroyWindow("capture frame")
            cv2.destroyWindow("Press ESC to captured initial position for tracking")
            break   
        
    plt.imshow(img_base_frame)
    ref = plt.ginput(NrPoints)
    #print(ref)           
    plt.close()


    # thickness of the bounding box lines drawn on the image
    thickness = 2
    # ground truth location drawn in green
    ground_truth_color = (0, 255, 0)
    # tracker location drawn in red
    result_color = (0, 0, 255)
    
    """
    ret, init_img = cap.read()
    if not ret:
        print("Initial frame could not be read")
        sys.exit(0)
    """
    
    # extract the true corners in the first frame and place them into a 2x4 array
    init_corners = [list(ref[0]),
                    list(ref[1]),
                    list(ref[2]),
                    list(ref[3])]

    # X refers to the initial corners selected
    X = np.ones((3, NrPoints))
    # X_P refers to the updated corners
    # X_P = np.ones((3, NrPoints))
    for i in range(NrPoints):
        X[:2, i] = ref[i]
    cover_homography = find_homography(cover, X, NrPoints, norm='euclidean',normalization=True)
    print("cover_homography",cover_homography)
    init_corners = np.array(init_corners).T
    # write the initial corners to the result file
    
    # initialize tracker with the first frame and the initial corners
    initTracker(init_img, init_corners)

    model = cv2.imread('ar_tag.png', 0)
    model = 1

    test_projection = projection_matrix(camera_parameters, cover_homography)
    image_base_frame_copy = np.copy(img_base_frame)    
    test_frame = render_with_bar_param(image_base_frame_copy, obj, test_projection, template_type, False)
    if args.test_param == 1:
        while True:
            param_testing_win_name = "Experienment for the params, press space button to see the result,press ESC to start live AR rendering"
            cv2.imshow(param_testing_win_name, test_frame)
            
            key = cv2.waitKey(5)
            if key == 32:
                #img_base_frame = cv2.imread(str(path))
                image_base_frame_copy = np.copy(img_base_frame)  
                test_frame = render_with_bar_param(image_base_frame_copy, obj, test_projection, model, False)
            elif key == 27:
                cv2.destroyWindow(param_testing_win_name)
             
                break
        

    window_name = 'Live rendering'
    cv2.namedWindow(window_name)

    # lists for accumulating the tracking error and fps for all the frames
    #tracking_fps = []

    #frame_id = 0
    cap = cv2.VideoCapture(camera_id)
    cv2.namedWindow("capture frame") 
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_FPS, 60) 
    
    while True:
        ret, src_img = cap.read()
        if not ret:
            print("failed to capture the image")
            return 
        # update the tracker with the current frame
        tracker_corners = updateTracker(src_img)

        X_P = np.ones((3, NrPoints))
        for i in range(NrPoints):
            X_P[:2, i] = tracker_corners[:, i]

        tracker_homography = find_homography(X, X_P, NrPoints, norm="euclidean",normalization=True)
        
        #print("tracker_homography",tracker_homography)
        overall_homography = tracker_homography.dot(cover_homography)


        # compute the tracking error
        if overall_homography is not None:
            pass

        # obtain 3D projection matrix from homography matrix and camera parameters
        projection = projection_matrix(camera_parameters, overall_homography)
        # project cube or model
        frame = render_with_bar_param(src_img, obj, projection, model, False)
        
        # draw the tracker location
        # write statistics (error and fps) to the image
        # display the image
        center_line_color = (0, 255, 255)
        if show_planar_tracking:
            drawRegion(src_img, tracker_corners, result_color, thickness)
        
        #cv2.imwrite('./image_out/src_img%05d.jpg' % frame_id, src_img)
        #warped_cover = apply_homography(book_cover, src_img,overall_homography, fit_origin=True, get_image = True)
        #cv2.imwrite('./image_out/warped_cover%05d.jpg' % frame_id, warped_cover)
        
        #added_image = cv2.addWeighted(src_img,0.6,warped_cover,0.7,0)
        cv2.imshow(window_name, frame)                        

        
        key = cv2.waitKey(1)#pauses for 3 seconds before fetching next image
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break            

def render_with_bar_param(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    rati, sca, x_rot, y_rot, z_rot, pr_down, t_x, t_y, t_z, r,g,b = read_bar_info()
    
    
    
    rot_mat_x = AnglesToRotationMatrix("x",x_rot)
    rot_mat_y = AnglesToRotationMatrix("y",y_rot)
    rot_mat_z = AnglesToRotationMatrix("z",z_rot)
    
    
    vertices = obj.vertices
    scale_matrix = np.eye(3) *10 * sca/100
    
    model_const = 300
    
    #case1: the surface is a square
    if type(model) is int:
        if model == 1:
            h = model_const
            w = model_const
        #case2: the surface is rectangle with hight greater than width
        elif model == 2:
            h = model_const * 1.5
            w = model_const
            
        elif model == 3:
            h = model_const
            w = model_const * 1.5
    else :
        h, w = model.shape
        h = h
        w = w
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        
        points = np.dot(points, scale_matrix)
        points = np.dot(points, rot_mat_x)
        points = np.dot(points, rot_mat_y)
        points = np.dot(points, rot_mat_z)
        
        points += np.array([t_x,t_y,t_z]).reshape(1,3)
        
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        
        #print("dst",dst)
        imgpts = np.int32(dst)
        
        if color is False:
            # color_buffer = randint(0, 256)
            # random_color = (color_buffer, color_buffer, color_buffer)
            color = (r,g,b)         
            #cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
            #color = np.array([r,g,b]) 
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
            
            cv2.polylines(img, imgpts, isClosed=True, color=0, thickness=2,
                          lineType=cv2.LINE_4)              
        else:
            #color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            color = (r,g,b)
            cv2.fillConvexPoly(img, imgpts, color)
            cv2.polylines(img, imgpts, isClosed=True, color=0.8, thickness= 2,
                          lineType=cv2.LINE_4)            
    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    
    input homography matrix will be 3 by 3
    """
    # Compute rotation along the x and y axis as well as the translation
    if homography[0,0] > 0:
        homography = homography * (-1)
    
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    

    #print("projection matrix:\n", projection)
    return np.dot(camera_parameters, projection)


#def projection_matrix(camera_parameters, homography):

    ##From the camera calibration matrix and the estimated homography
    ##compute the 3D projection matrix
    
    ##input homography matrix will be 3 by 3

    ## Compute rotation along the x and y axis as well as the translation
    #if homography[0,0] > 0:
        #homography = homography * (-1)
    
    
    #H = np.dot(np.linalg.inv(camera_parameters), homography)
    
    
    #H1t = H[:, 0]
    #H2t = H[:, 1]
    #H3t = H[:, 2]
    #H1t_norm = np.linalg.norm(H1t, 2)
    #H2t_norm = np.linalg.norm(H2t, 2)
    
    #R1t = H1t/H1t_norm
    #R2t = H2t/H2t_norm
    #R3t = np.cross(R1t,R2t)
    #Tt = 2*H3t/(H1t_norm + H2t_norm)
    ## normalise vectors

    ## finally, compute the 3D projection matrix from the model to the current frame
    #projection = np.stack((R1t, R2t, R3t, Tt)).T

    ##print("projection matrix:\n", projection)
    #return np.dot(camera_parameters, projection)


def read_bar_info():
    rati = cv2.getTrackbarPos(ratio_bar, barsWindow)
    sca = cv2.getTrackbarPos(scale_bar, barsWindow)
    x_rot = cv2.getTrackbarPos(x_bar, barsWindow)
    y_rot = cv2.getTrackbarPos(y_bar, barsWindow)
    z_rot = cv2.getTrackbarPos(z_bar, barsWindow)
    pr_down = cv2.getTrackbarPos(pointDir_bar, barsWindow)
    t_x = cv2.getTrackbarPos(t_1, barsWindow) - 50
    t_y = cv2.getTrackbarPos(t_2, barsWindow) - 50
    t_z = cv2.getTrackbarPos(t_3, barsWindow) - 50
    r_c = cv2.getTrackbarPos(r, barsWindow)
    g_c = cv2.getTrackbarPos(g, barsWindow)
    b_c = cv2.getTrackbarPos(b, barsWindow)
    
    return(rati,sca,x_rot,y_rot,z_rot,pr_down,t_x,t_y,t_z,r_c,g_c,b_c)



def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def AnglesToRotationMatrix(axis,theta_angle):
    theta_radian = theta_angle * np.pi/180
    
    if axis == "x":
        R = np.array([[1, 0, 0],
                        [0, np.cos(theta_radian), -np.sin(theta_radian)],
                        [0, np.sin(theta_radian), np.cos(theta_radian)]
                        ])        
    elif axis == "y":   
        R = np.array([[np.cos(theta_radian), 0, np.sin(theta_radian)],
                    [0, 1, 0],
                    [-np.sin(theta_radian), 0, np.cos(theta_radian)]
                    ])
    else:
        R = np.array([[np.cos(theta_radian), -np.sin(theta_radian), 0],
                    [np.sin(theta_radian), np.cos(theta_radian), 0],
                    [0, 0, 1]
                    ])
    return R

if __name__ == '__main__':
    main()