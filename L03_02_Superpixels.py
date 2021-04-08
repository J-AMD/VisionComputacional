import numpy as np
import cv2 as cv

img = cv.imread('imagenes\\sudoku.png')


slic = cv.ximgproc.createSuperpixelSLIC(img, region_size=100, ruler = 20.0)

slic.iterate(10)
mask_slic = slic.getLabelContourMask()
label_slic = slic.getLabels()
number_slic = slic.getNumberOfSuperpixels()
mask_inv_slic = cv.bitwise_not(mask_slic)

img_slic = cv.bitwise_and(img, img, mask=mask_inv_slic)

cv.imshow("img_slic",img_slic)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite('SPS.jpg', img)