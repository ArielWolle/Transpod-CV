import cv2
from cv2 import line
import numpy as np
import math
import matplotlib.pyplot as plt

#width and length of all imagesx
width=int(640)
length=int(480)

image_name="IMG_3912.JPG"
mask_image_name="mask.jpg"
output_image=None

use_mask=False
use_color_mask=True
use_cap1=True

use_color_settings=True
use_line_settings=True
use_circle_settings=True

#set up line trackbars

open_trackbar=False

#circle parameters
dp=0.6
minDist=100

#canny edge detection parameters
low_threshold = 30
high_threshold = 150

#orange detection parameters
ORANGE_MIN = np.array([0, 38, 118])
ORANGE_MAX = np.array([9, 255, 255])

#line parameters
rho = 1  # distance resolution in pixels of the Hough grid
theta = 200  # angular resolution in radians of the Hough grid
threshold = 20  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 125  # minimum number of pixels making up a line
max_line_gap = 25  # maximum gap in pixels between connectable line segments

#trackbar setup
def nothing(x):
    pass


cap1_shape=0
cap2_shape=0



cap1 = cv2.VideoCapture(0)
if not (cap1.isOpened()):
    print("Could not open onboard video device")
else:
    ret2, frame2 = cap1.read() 
    cap1_shape=frame2.shape
    print(cap1_shape)
cap2 = cv2.VideoCapture('rtsp://172.20.10.10:8554/mjpeg/1')
if not (cap2.isOpened()):
    print("Could not open USB video device")
else:
    ret2, frame2 = cap2.read() 
    cap2_shape=frame2.shape
    print(cap2_shape)

#enable color trackbars
def enable_color_trackbars():
    cv2.namedWindow('color_trackbars')
    # Create color_trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar('HMin', 'color_trackbars', 0, 179, nothing)
    cv2.createTrackbar('HMax', 'color_trackbars', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'color_trackbars', 0, 255, nothing)        
    cv2.createTrackbar('SMax', 'color_trackbars', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'color_trackbars', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'color_trackbars', 0, 255, nothing)

    # Set default value for Min and Max HSV trackbars
    cv2.setTrackbarPos('HMin', 'color_trackbars', ORANGE_MIN[0])
    cv2.setTrackbarPos('SMin', 'color_trackbars', ORANGE_MIN[1])
    cv2.setTrackbarPos('VMin', 'color_trackbars', ORANGE_MIN[2])
    cv2.setTrackbarPos('HMax', 'color_trackbars', ORANGE_MAX[0])
    cv2.setTrackbarPos('SMax', 'color_trackbars', ORANGE_MAX[1])
    cv2.setTrackbarPos('VMax', 'color_trackbars', ORANGE_MAX[2])

#enable line trackbars
def enable_line_trackbars():
    cv2.namedWindow('line_trackbars')
    cv2.createTrackbar('Low Threshold', 'line_trackbars', 0, 255, nothing)
    cv2.createTrackbar('High Threshold', 'line_trackbars', 0, 255, nothing)
    cv2.createTrackbar('rho', 'line_trackbars', 0, 1000, nothing)
    cv2.createTrackbar('theta', 'line_trackbars', 1, 1000, nothing)
    cv2.createTrackbar('threshold', 'line_trackbars', 0, 1000, nothing)
    cv2.createTrackbar('min_line_length', 'line_trackbars', 0, 1000, nothing)
    cv2.createTrackbar('max_line_gap', 'line_trackbars', 0, 1000, nothing)

    

    cv2.setTrackbarPos('Low Threshold', 'line_trackbars', low_threshold)
    cv2.setTrackbarPos('High Threshold', 'line_trackbars', high_threshold)
    cv2.setTrackbarPos('rho', 'line_trackbars', rho)
    cv2.setTrackbarPos('theta', 'line_trackbars', theta)
    cv2.setTrackbarPos('threshold', 'line_trackbars', threshold)
    cv2.setTrackbarPos('min_line_length', 'line_trackbars', min_line_length)
    cv2.setTrackbarPos('max_line_gap', 'line_trackbars', max_line_gap)

def enable_circle_trackbars():
    cv2.namedWindow('circle_trackbars')
    cv2.createTrackbar('minDist', 'circle_trackbars', 0, 1000, nothing)
    cv2.createTrackbar('dp', 'circle_trackbars', 0, 1000, nothing)
            

    cv2.setTrackbarPos('minDist', 'circle_trackbars', minDist)
    cv2.setTrackbarPos('dp', 'circle_trackbars', int(dp*100))
#show multple images in one window
def show(imgs, win="Image", scale=1):
    imgs = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) \
            if len(img.shape) == 2 \
            else img for img in imgs]
    img_concat = np.concatenate(imgs, 1)
    h, w = img_concat.shape[:2]
    cv2.imshow(win, cv2.resize(img_concat, (int(w * scale), int(h * scale))))

def detect(input_image):
    #take in input and mask images
    input_image=cv2.resize(input_image,(width,length))
    #if we are using the mask, we need to resize the mask to fit the image
    if use_mask:
        mask_image=cv2.imread(mask_image_name)
        mask_image=cv2.resize(mask_image,((width,length)))
        mask = cv2.inRange(mask_image, np.array([0, 0, 0]), np.array([15, 15, 15]))

    #limit it to certain colours
    if use_color_mask:
        
        hsv_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2HSV)
        mask_colour=cv2.inRange(hsv_image, ORANGE_MIN, ORANGE_MAX)
        cv2.imshow("line_image",mask_colour)

    #set output image to input image
    output_image=input_image.copy()

    #seperate into 2 streams the circle and the line
    circle_image=cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    line_image=input_image.copy()
    if use_mask:
        line_image=cv2.bitwise_and(line_image, line_image, mask=mask)
    if use_color_mask:
        #line_image=cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        line_image=mask_colour
    else:
        line_image=cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    #blur both images
    circle_image=cv2.GaussianBlur(circle_image, (5,5), 0)
    line_image=cv2.GaussianBlur(line_image, (15,15), 0)


    #find the circles
    circles = cv2.HoughCircles(circle_image, cv2.HOUGH_GRADIENT, dp, minDist)
    if circles is not None:
    # Get the (x, y, r) as integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the circles
        for (x, y, r) in circles:
            cv2.circle(output_image, (x, y), r, (255, 255, 0), 2)

    #find the lines

    #edge detection
    edges = cv2.Canny(line_image, low_threshold, high_threshold)

    lines = cv2.HoughLinesP(edges, rho,np.pi / theta, threshold, np.array([]),min_line_length, max_line_gap)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return output_image

ret=None
frame=None
while(True):
    if open_trackbar:
        if use_color_settings:
            try:
                hMin = cv2.getTrackbarPos('HMin', 'color_trackbars')
                sMin = cv2.getTrackbarPos('SMin', 'color_trackbars')
                vMin = cv2.getTrackbarPos('VMin', 'color_trackbars')
                hMax = cv2.getTrackbarPos('HMax', 'color_trackbars')
                sMax = cv2.getTrackbarPos('SMax', 'color_trackbars')
                vMax = cv2.getTrackbarPos('VMax', 'color_trackbars')
                ORANGE_MIN = np.array([hMin, sMin, vMin])
                ORANGE_MAX = np.array([hMax, sMax, vMax])  
            except Exception as e:
                use_color_settings=False
        if use_line_settings:
            try:
                rho = cv2.getTrackbarPos('rho', 'line_trackbars')
                theta = cv2.getTrackbarPos('theta', 'line_trackbars')
                threshold = cv2.getTrackbarPos('threshold', 'line_trackbars')
                min_line_length = cv2.getTrackbarPos('min_line_length', 'line_trackbars')
                max_line_gap = cv2.getTrackbarPos('max_line_gap', 'line_trackbars')
            except Exception as e:
                use_line_settings=False
        if use_circle_settings:
            try:
                dp = cv2.getTrackbarPos('dp', 'circle_trackbars')/100
                minDist = cv2.getTrackbarPos('minDist', 'circle_trackbars')
            except Exception as e:
                use_circle_settings=False

    # Capture frame-by-frame  
    
    if use_cap1:
        ret, frame = cap1.read() 
    elif not use_cap1:
        ret, frame = cap2.read()
    
    show([frame,detect(frame)], "Images", 1)
    key=cv2.waitKey(10)& 0xFF
    if key == ord('c'):    
        if use_cap1 and cap2_shape:
            width=cap2_shape[1]
            length=cap2_shape[0]
            use_cap1=False
        elif not use_cap1 and cap1_shape:
            width=cap1_shape[1]
            length=cap1_shape[0]
            use_cap1=True
    elif key == ord('1') and cap1_shape:
        width=cap1_shape[1]
        length=cap1_shape[0]
        use_cap1=True
    elif key == ord('2') and cap2_shape:
        width=cap2_shape[1]
        length=cap2_shape[0]
        use_cap1=False

    elif key == ord('3'):
        use_color_settings=not use_color_settings       
    elif key == ord('4'):
        use_line_settings=not use_line_settings
    elif key == ord('5'):
        use_circle_settings=not use_circle_settings
    elif key == ord('s'):
        if open_trackbar:
            open_trackbar=False
            if use_color_settings:
                cv2.destroyWindow('color_trackbars')
            if use_line_settings:
                cv2.destroyWindow('line_trackbars')
            if use_circle_settings:
                cv2.destroyWindow('circle_trackbars')
        else:
            open_trackbar=True
            # Create a window
            if use_circle_settings:
                enable_color_trackbars()
            if use_line_settings:
                enable_line_trackbars()
            if use_circle_settings:
                enable_circle_trackbars()
    elif key == ord('q'):   
            break

cap1.release()
cap2.release()
cv2.destroyAllWindows()