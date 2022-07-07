import cv2
import numpy as np

def show(imgs, win="Image", scale=1):
    imgs = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) \
            if len(img.shape) == 2 \
            else img for img in imgs]
    img_concat = np.concatenate(imgs, 1)
    h, w = img_concat.shape[:2]
    cv2.imshow(win, cv2.resize(img_concat, (int(w * scale), int(h * scale))))

d = {"Hue Min": (0, 179),
     "Hue Max": (116, 179),
     "Sat Min": (0, 255),
     "Sat Max": (30, 255),
     "Val Min": (160, 255),
     "Val Max": (253, 255),
     "k1": (31, 50),
     "k2": (31, 50),
     "sigma": (10, 20)}

img = cv2.imread(r"IMG_3891.JPG")
cv2.namedWindow("Track Bars",cv2.WINDOW_NORMAL)

for i in d:
    cv2.createTrackbar(i, "Track Bars", *d[i], id)

img = cv2.imread("IMG_3891.JPG")

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
while True:
    h_min, h_max, s_min, s_max, v_min, v_max, k1, k2, s = (cv2.getTrackbarPos(i, "Track Bars") for i in d)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(img_hsv, lower, upper)
    mask = cv2.erode(mask, np.ones((3, 3)))
    k1, k2 = k1 // 2 * 2 + 1, k2 // 2 * 2 + 1
    img_blurred = cv2.GaussianBlur(img, (k1, k2), s)
    result = img_blurred.copy()
    result[mask == 0] = img[mask == 0]
    show([img, mask], "Window 1", 0.5) # Show original image & mask
    show([img_blurred, result], "Window 2", 0.5) # Show blurred image & result
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break