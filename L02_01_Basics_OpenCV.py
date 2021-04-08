import numpy as np
import cv2

# Crear matriz
img = np.zeros((3, 3), dtype=np.uint8)


# leer imagen
img = cv2.imread('DataSet/Lenna.png')
type(img)
img.shape
img.size
img.dtype


# Visualizar
def imshow(img):
    cv2.imshow('Ventana', img)
    cv2.waitKey()

imshow(img)

# Escribir
cv2.imwrite('save.jpg', img)

# leer imagen
img2 = cv2.imread('DataSet/Lenna.png', cv2.IMREAD_GRAYSCALE)
imshow(img2)

# Mostrar cada canal
img_temp = np.zeros(img.shape, dtype=np.uint8)
img_temp [:,:,0] = img[:,:,0]
imshow(img_temp)

img_temp = np.zeros(img.shape, dtype=np.uint8)
img_temp [:,:,1] = img[:,:,1]
imshow(img_temp)

img_temp = np.zeros(img.shape, dtype=np.uint8)
img_temp [:,:,2] = img[:,:,2]
imshow(img_temp)

# Región de interés
roi = img[0:100, 0:100]
imshow(roi)

# -----------------------------------------------------------------------------------------------------
# Reproducir video:
videoCapture = cv2.VideoCapture('MyInputVid.avi')
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('MyOutputVid.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

success, frame = videoCapture.read()
while success:
    videoWriter.write(frame)
    success, frame = videoCapture.read()
# -----------------------------------------------------------------------------------------------------

# Leer cámara
import cv2
cameraCapture = cv2.VideoCapture(0)
fps = 30 # An assumption
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('MyOutputVid.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)
success, frame = cameraCapture.read()
numFramesRemaining = 10 * fps - 1 # 10 seconds of frames
while success and numFramesRemaining > 0:
    videoWriter.write(frame)
    success, frame = cameraCapture.read()
    numFramesRemaining -= 1
# -----------------------------------------------------------------------------------------------------

# Canny
import cv2
import numpy as np
img = cv2.imread('DataSet/Lenna.png', 0)
cv2.imwrite("canny.jpg", cv2.Canny(img, 150, 180))
cv2.imshow("canny", cv2.imread("canny.jpg"))
cv2.waitKey()
cv2.destroyAllWindows()
# -----------------------------------------------------------------------------------------------------