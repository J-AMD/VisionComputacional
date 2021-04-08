# Detector de esquinas de Harris-------------------------------------------------------------
import cv2
import numpy as np

img = cv2.imread('imagenes\\imagen.jpeg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img = np.float32(gray_img)

dst = cv2.cornerHarris(gray_img, blockSize=2, ksize=3, k=0.04)

# dilate to mark the corners
dst = cv2.dilate(dst, None)
img[dst > 0.01 * dst.max()] = [0, 255, 0]

cv2.imshow('haris_corner', img)
cv2.waitKey()
#-------------------------------------------------------------------------------------------------

# Detector Shi-Tomasi ----------------------------------------------------------------------------
import cv2
import numpy as np

img = cv2.imread('imagenes\\imagen.jpeg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray_img, maxCorners=50, qualityLevel=0.02, minDistance=20)
corners = np.float32(corners)

for item in corners:
    x, y = item[0]
    cv2.circle(img, (x, y), 6, (0, 255, 0), -1)

cv2.imshow('good_features', img)
cv2.waitKey()
#-------------------------------------------------------------------------------------------------


# SIFT -------------------------------------------------------------------------------------------
# Nota importante: Solo funciona en versiones anteriores a 3.4.2.16
# pip install opencv-python==3.4.2.16
# pip install opencv-contrib-python==3.4.2.16

import cv2

img = cv2.imread('imagenes\\imagen.jpeg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray_img, None)

kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0),
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SIFT', kp_img)
cv2.waitKey()
#-------------------------------------------------------------------------------------------------

# SURF -------------------------------------------------------------------------------------------
import cv2

img = cv2.imread('imagenes\\imagen.jpeg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create()
kp, des = surf.detectAndCompute(gray_img, None)

kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SURF', kp_img)
cv2.waitKey()
#-------------------------------------------------------------------------------------------------


# ORB --------------------------------------------------------------------------------------------
import cv2

img = cv2.imread('imagenes\\imagen.jpeg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=2000)
kp, des = orb.detectAndCompute(gray_img, None)

kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

cv2.imshow('ORB', kp_img)
cv2.waitKey()
#-------------------------------------------------------------------------------------------------
