from mpl_toolkits import mplot3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio


cam = cv2.VideoCapture(0)
cv2.namedWindow("capture frame")
img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("frame captured", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ASCII:ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # ASCII:SPACE pressed
        img_name = "%05d.jpg"%img_counter
        img_name = "%05d.jpg"%img_counter
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
