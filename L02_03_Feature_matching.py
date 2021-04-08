import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Read Images
img1 = cv.imread('imagenes\\obj1.jpeg', cv.IMREAD_GRAYSCALE) # queryImage
img2 = cv.imread('imagenes\\obj2.jpeg', cv.IMREAD_GRAYSCALE) # trainImage

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x:x.distance)

# Draw matches
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches, None)

plt.imshow(img3), plt.show()

