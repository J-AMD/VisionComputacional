import numpy as np
import cv2 as cv
import random

image = cv.imread('DataSet/water_coins.jpg')

ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)

ss.switchToSelectiveSearchQuality()
rects = ss.process()

for i in range(0, len(rects), 100):
    # clone the original image so we can draw on it
    output = image.copy()
    # loop over the current subset of region proposals
    for (x, y, w, h) in rects[i:i + 100]:
        # draw the region proposal bounding box on the image
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv.rectangle(output, (x, y), (x + w, y + h), color, 2)

    cv.imshow("Output", output)
    key = cv.waitKey(0) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
