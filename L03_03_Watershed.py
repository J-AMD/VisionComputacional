import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('imagenes/water_vestido2.png')

cv.imshow('ventana', img)
cv.waitKey()

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('ventana', gray_img)
cv.waitKey()

ret, thresh = cv.threshold(gray_img, 0, 255,
                           cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

cv.imshow('ventana', thresh)
cv.waitKey()

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv.dilate(opening, kernel, iterations=3)

cv.imshow('ventana', sure_bg)
cv.waitKey()

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
cv.normalize(dist_transform, dist_transform, 0, 1.0, cv.NORM_MINMAX);

cv.imshow('ventana', dist_transform)
cv.waitKey()

ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

cv.imshow('ventana', sure_fg)
cv.waitKey()

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

cv.imshow('ventana', unknown)
cv.waitKey()

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers = cv.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

cv.imshow('ventana', img)
cv.waitKey()






