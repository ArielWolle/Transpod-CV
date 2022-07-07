
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

check_intersection=5
slope_threshold=0

def show(imgs, win="Image", scale=1):
    imgs = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) \
            if len(img.shape) == 2 \
            else img for img in imgs]
    img_concat = np.concatenate(imgs, 1)
    h, w = img_concat.shape[:2]
    cv2.imshow(win, cv2.resize(img_concat, (int(w * scale), int(h * scale))))

def slope(x1, y1, x2, y2): # Line slope given two points:
    try:
        return (y2 - y1) / (x2 - x1)
    except ZeroDivisionError:
        return 99999

def angle(s1, s2): 
    return math.degrees(math.atan((s2-s1)/(1+(s2*s1))))
width=int(240*1.5)
length=int(320*1.5)
image = cv2.imread('IMG_3898.JPG')
image=cv2.resize(image,(width,length))
mask_image = cv2.imread("mask.jpg")
mask_image=cv2.resize(mask_image,((width,length)))
print(image.shape)
#resize mask to fit image size
mask = cv2.inRange(mask_image, np.array([0, 0, 0]), np.array([15, 15, 15]))
mask_colour=cv2.inRange(image, np.array([30,30, 30]), np.array([140, 140, 140]))

#cv2.imshow("masked",masked)
output = image.copy()
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
line_image = cv2.bitwise_and(image, image, mask=mask)
line_image_colour = cv2.bitwise_and(line_image, line_image, mask=mask_colour)
line_image_blur=cv2.GaussianBlur(line_image_colour, (5, 5), 0)
img = cv2.GaussianBlur(img, (7, 7), 0)


#find edges
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(line_image_blur, low_threshold, high_threshold)
show([edges,line_image_blur,line_image_colour,mask],"lines mask",1)

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 200  # angular resolution in radians of the Hough grid
threshold = 20  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 150  # minimum number of pixels making up a line
max_line_gap = 40  # maximum gap in pixels between connectable line segments

lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


# Find circles
cv2.imshow('image', img)
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 0.25, 100)
# If some circle is found
if circles is not None:
   # Get the (x, y, r) as integers
   circles = np.round(circles[0, :]).astype("int")
   # loop over the circles
   for (x, y, r) in circles:
      cv2.circle(output, (x, y), r, (255, 255, 0), 2)
points = []

instersection_list=[]
if lines is not None:
    lines = lines.tolist()
    for line in lines:
        intersection=0
        for x1, y1, x2, y2 in line:
            if abs(angle(slope(x1, y1, x2, y2),99999))>slope_threshold:
                if check_intersection:
                    for line_s in lines[-lines.index(line):]:
                        for x3, y3, x4, y4 in line_s:
                            if intersect([x1, y1], [x2, y2], [x3, y3], [x4, y4]):
                                intersection+=1
                    instersection_list.append(intersection)
                    if intersection<check_intersection:
                        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                print(angle(slope(x1, y1, x2, y2),99999))
if check_intersection:
    print(max(instersection_list))

# setting RGB color list:
color = ('blue', 'green', 'red')

# Iterating throuhg each channel and plotting the corresponding result:
# using cv.calcHist() opencv method
for i,color in enumerate(color):
    histogram = cv2.calcHist([line_image_blur], [i], None, [256], [1, 256])
    cdf = histogram.cumsum()
    cdf_percent = cdf / cdf.max()
    plt.plot(histogram, color=color, label=color+'_channel')
    # plt.plot(cdf_percent, color=color, label=color+'_cdf')
    plt.xlim([0,256])

plt.title('Histogram Analysis',fontsize=20)
plt.xlabel('Range intensity values',fontsize=14)
plt.ylabel('Count of Pixels',fontsize=14)
plt.legend()
plt.show()
cv2.imshow("circle",output)
cv2.waitKey(0)