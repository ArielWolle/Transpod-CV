import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img_rgb = cv.imread('IMG_3894.JPG')
img_rgb=cv.resize(img_rgb,(480,640))
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('needle.jpg',0)
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
res_8=(res*255).astype(np.uint8)
ret,res_thresh = cv.threshold(res_8,200,255*1,cv.THRESH_BINARY)
res_blur=(cv.GaussianBlur(res_thresh,(5,5),0))

edges = cv.Canny(res_blur, 50, 150)

lines = cv.HoughLinesP(edges, 1,np.pi / 200, 20, np.array([]),100, 15)

if lines is not None:
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv.imshow('res',res_blur)
cv.imshow("bera",res_8)
cv.imshow('edges',edges)
cv.imshow("hello",res)
threshold = 0.7
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
cv.imshow('res.png',img_rgb)
cv.waitKey()
cv.destroyAllWindows()