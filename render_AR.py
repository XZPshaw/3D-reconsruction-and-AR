"""
Todo1: user can choose ratio of the template/cover and set length ?
Todo2: Reconstruction part

"""



import os
import sys
import cv2
import numpy as np
import math
import time
import re

from matplotlib import pyplot as plt
from q3flag_finalized import find_homography, apply_homography
from objloader_simple import *

DEFAULT_COLOR = (0, 0, 0)


def nothing(x):
    pass


def nothing(x):
    pass

# named ites for easy reference
barsWindow = 'Bars'
ratio_bar = 'ratio'
scale_bar = 'scale(%)'
x_bar = 'ratation_x'
y_bar = 'ratation_y'
z_bar = 'ratation_z'
t_1 = "t_x"
t_2 = "t_y"
t_3 = "t_z"

pointDir_bar = 'V High'


# create window for the slidebars
cv2.namedWindow(barsWindow, flags = cv2.WINDOW_NORMAL)

# create the sliders
cv2.createTrackbar(ratio_bar, barsWindow, 0, 5, nothing)
cv2.createTrackbar(scale_bar, barsWindow, 0, 40000, nothing)
cv2.createTrackbar(x_bar, barsWindow, 0, 360, nothing)
cv2.createTrackbar(y_bar, barsWindow, 0, 360, nothing)
cv2.createTrackbar(z_bar, barsWindow, 0, 360, nothing)
cv2.createTrackbar(pointDir_bar, barsWindow, 0, 1, nothing)
cv2.createTrackbar(t_1, barsWindow, -100, 100, nothing)
cv2.createTrackbar(t_2, barsWindow, -100, 100, nothing)
cv2.createTrackbar(t_3, barsWindow, -100, 100, nothing)

# set initial values for sliders
cv2.setTrackbarPos(ratio_bar, barsWindow, 1)
cv2.setTrackbarPos(scale_bar, barsWindow, 100)
cv2.setTrackbarPos(x_bar, barsWindow, 0)
cv2.setTrackbarPos(y_bar, barsWindow, 0)
cv2.setTrackbarPos(z_bar, barsWindow, 0)
cv2.setTrackbarPos(pointDir_bar, barsWindow, 0)
cv2.setTrackbarPos(t_1, barsWindow, 0)
cv2.setTrackbarPos(t_2, barsWindow, 0)
cv2.setTrackbarPos(t_3, barsWindow, 0)

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


def readTrackingData(filename):
    if not os.path.isfile(filename):
        print("Tracking data file not found:\n ", filename)
        sys.exit()

    data_file = open(filename, 'r')
    lines = data_file.readlines()
    no_of_lines = len(lines)
    data_array = np.zeros((no_of_lines, 8))
    line_id = 0
    for line in lines:
        words = line.split()[1:]
        if len(words) != 8:
            msg = "Invalid formatting on line %d" % line_id + " in file %s" % filename + ":\n%s" % line
            raise SyntaxError(msg)
        coordinates = []
        for word in words:
            coordinates.append(float(word))
        data_array[line_id, :] = coordinates
        line_id += 1
    data_file.close()
    return data_array


def writeCorners(file_id, corners):
    # write the given corners to the file
    corner_str = ''
    for i in range(4):
        corner_str = corner_str + '{:5.2f}\t{:5.2f}\t'.format(corners[0, i], corners[1, i])
    file_id.write(corner_str + '\n')


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
    lk_params = dict(winSize=(32, 32),
                     maxLevel=8,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 9, 0.02555))
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame_img, p0, None, **lk_params)
    old_frame = frame_img.copy()
    p0 = p1.copy()

    print(p1.T)
    return p1.T


def main():
    """
    addded part for model projection preparation

    """
    camera_parameters = np.array([[1430, 0, 480], [0, 1430, 620], [0, 0, 1]])
    #camera_parameters = np.array([[715, 0, 480], [0, 715, 620], [0, 0, 1]])
    # Load 3D model from OBJ file
    #obj = OBJ('models/fox.obj', swapyz=True)
    # obj = OBJ('low-poly-fox-by-pixelmannen.obj', swapyz=True,texture_file='texture.png')
    # obj = OBJ('./Pix2Vox-master/result/test/model.obj', swapyz=True,texture_file='texture.png')
    obj = OBJ('./Pix2Vox-master/result/test/model.obj', swapyz=True)
    # obj = OBJ('low-poly-fox-by-pixelmannen.obj', swapyz=True,texture_file= None)
    
    ################## if apply ar to existing video

    """
    end of this part
    """

    sequences = ['input_scene']
    seq_id = 0

    write_stats_to_file = 0
    show_tracking_output = 1

    seq_name = sequences[seq_id]
    print('seq_id: ', seq_id)
    print('seq_name: ', seq_name)

    #src_fname = seq_name + '/frame%05d.jpg'
    src_fname = seq_name + '/%05d.jpg'
    result_fname = seq_name + '_res.txt'

    path_img = './input_scene'
    ft = 'jpg'
    # ft = 'png'
    imgs = loadImgs_plus(path_img, ft, grayscale=False)
    print("imgs shape:", imgs.shape)

    NrPoints = 4
    # read the ground truth

    cover_path = 'pure_bookCover.jpg'
    book_cover = cv2.imread(str(cover_path))
    plt.imshow(book_cover)

    ref_cover = np.array([[0,0],[book_cover.shape[1] - 1, 0],[book_cover.shape[1] - 1, book_cover.shape[0] - 1],[0, book_cover.shape[0] - 1]])
    
    """
    ref_cover = plt.ginput(NrPoints)
    """
    cover = np.ones((3, NrPoints))
    

    for i in range(NrPoints):
        cover[:2, i] = ref_cover[i]

    # path = seq_name + '/frame%05d.jpg' % 1
    path = seq_name + '/%05d.jpg' % 0
    no_of_frames = imgs.shape[0]
    img_base_frame = cv2.imread(str(path))
    print(path)
    plt.imshow(img_base_frame)
    ref = plt.ginput(NrPoints)
    print(ref)

    result_file = open(result_fname, 'w')

    cap = cv2.VideoCapture()
    if not cap.open(src_fname):
        print('The video file ', src_fname, ' could not be opened')
        sys.exit()

    # thickness of the bounding box lines drawn on the image
    thickness = 2
    # ground truth location drawn in green
    ground_truth_color = (0, 255, 0)
    # tracker location drawn in red
    result_color = (0, 0, 255)

    print('no_of_frames: ', no_of_frames)

    ret, init_img = cap.read()
    if not ret:
        print("Initial frame could not be read")
        sys.exit(0)

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
    cover_homography = find_homography(cover, X, NrPoints, norm='euclidean',normalization=False)
    print("cover_homography",cover_homography)
    init_corners = np.array(init_corners).T
    # write the initial corners to the result file
    writeCorners(result_file, init_corners)
    
    #apply_homography(book_cover, img_base_frame,cover_homography, fit_origin=True)
    # initialize tracker with the first frame and the initial corners
    initTracker(init_img, init_corners)

    model = cv2.imread('reference/model.jpg', 0)

    test_projection = projection_matrix(camera_parameters, cover_homography)
    test_frame = render_with_bar_param(img_base_frame, obj, test_projection, model, False)
    
    while True:
        cv2.imshow("Experienment for the projected param", test_frame)
        
        key = cv2.waitKey(5)
        if key == 32:
            img_base_frame = cv2.imread(str(path))
            test_frame = render_with_bar_param(img_base_frame, obj, test_projection, model, False)
        elif key == 27:
            break
        
        
    if show_tracking_output:
        # window for displaying the tracking result
        window_name = 'Tracking Result'
        cv2.namedWindow(window_name)

    # lists for accumulating the tracking error and fps for all the frames
    tracking_errors = []
    tracking_fps = []

    for frame_id in range(1, no_of_frames):
        
        
        print("frame_id:",frame_id)
        ret, src_img = cap.read()
        if not ret:
            print("Frame ", frame_id, " could not be read")
            break

        start_time = time.process_time()
        """update the tracker with the current frame"""
        tracker_corners = updateTracker(src_img)
        #print(tracker_corners.shape)

        X_P = np.ones((3, NrPoints))
        for i in range(NrPoints):
            X_P[:2, i] = tracker_corners[:, i]

        tracker_homography = find_homography(X, X_P, NrPoints, norm="euclidean",normalization=False)
        
        print("tracker_homography",tracker_homography)
        overall_homography = tracker_homography.dot(cover_homography)
        end_time = time.process_time() + 1

        # write the current tracker location to the result text file
        writeCorners(result_file, tracker_corners)

        # compute the tracking fps
        current_fps = 1.0 / (end_time - start_time)
        tracking_fps.append(current_fps)

        # compute the tracking error

        if overall_homography is not None:
            pass

        # obtain 3D projection matrix from homography matrix and camera parameters
        projection = projection_matrix(camera_parameters, overall_homography)
        # project cube or model
        frame = render(src_img, obj, projection, model, False)
        
        if show_tracking_output:
            # draw the tracker location
            drawRegion(src_img, tracker_corners, result_color, thickness)
            # write statistics (error and fps) to the image
            # display the image
            center_line_color = (0, 255, 255)
            cv2.imwrite('./Q4Output/src_img%05d.jpg' % frame_id, src_img)
            
            warped_cover = apply_homography(book_cover, src_img,overall_homography, fit_origin=True, get_image = True)
            cv2.imwrite('./Q4Output/warped_cover%05d.jpg' % frame_id, warped_cover)
            
            added_image = cv2.addWeighted(src_img,0.6,warped_cover,0.7,0)
            cv2.imshow(window_name, added_image)                        
            #cv2.imshow(window_name, warped_cover)            
            cv2.imwrite('./Q4Output/added_image%05d.jpg' % frame_id, added_image)
            
            
            key = cv2.waitKey(3000)#pauses for 3 seconds before fetching next image
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows()
                break            
            #if cv2.waitKey(1) == 27:
                
                #break
                # print 'curr_error: ', curr_error

    result_file.close()


def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
        
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 2
    h, w = model.shape
    h = h
    w = w
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        
        
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def render_with_bar_param(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    rati,sca,x_rot,y_rot,z_rot,pr_down,t_x,t_y,t_z = read_bar_info()
    
    rot_mat_x = AnglesToRotationMatrix("x",x_rot)
    rot_mat_y = AnglesToRotationMatrix("y",y_rot)
    rot_mat_z = AnglesToRotationMatrix("z",z_rot)
    
    
    vertices = obj.vertices
    scale_matrix = np.eye(3) * sca/100
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
        # print("face_vertices:\n",points)
        
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
                
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
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

    print("projection matrix:\n", projection)
    return np.dot(camera_parameters, projection)



def read_bar_info():
    rati = cv2.getTrackbarPos(ratio_bar, barsWindow)
    sca = cv2.getTrackbarPos(scale_bar, barsWindow)
    x_rot = cv2.getTrackbarPos(x_bar, barsWindow)
    y_rot = cv2.getTrackbarPos(y_bar, barsWindow)
    z_rot = cv2.getTrackbarPos(z_bar, barsWindow)
    pr_down = cv2.getTrackbarPos(pointDir_bar, barsWindow)
    t_x = cv2.getTrackbarPos(t_1, barsWindow)
    t_y = cv2.getTrackbarPos(t_2, barsWindow)
    t_z = cv2.getTrackbarPos(t_3, barsWindow)
    return(rati,sca,x_rot,y_rot,z_rot,pr_down,t_x,t_y,t_z)



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