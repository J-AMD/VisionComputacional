import cv2
import numpy as np 

def Segmentacion(Threshold):
	vc2, contours, hierarchy = cv2.findContours(Threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#cv2.imshow("Contornos de THD", vc2)
#print("He encontrado {} objetos".format(len(contours)))
	cont=1
	for c in contours:
			if cv2.contourArea(c) < 1500:
				continue
 
			(x, y, w, h) = cv2.boundingRect(c)
	     
			cv2.rectangle(vc, (x, y), (x + w, y + h), (0, 255, 0), 2)
			crop_img = vc[y:y+h, x:x+w]
			cv2.imshow("cropped"+str(cont), crop_img)
			cv2.imwrite('C:\Users\gmsvuser\Documents\PythonExercise\Capture'+str(cont)+'.jpg', crop_img) #Guarda c/recorte en arcivo especifico
			cv2.waitKey(0)
			print cont
			#print c
			cont+=1
	cv2.imshow("Rectangulo", vc)

def CropSegmento(Segmento):
	#GrayCrop = cv2.cvtColor(Segmento, cv2.COLOR_BGR2GRAY)
	#retval, BinaryCrop = cv2.threshold(vc1, 140, 255, cv2.THRESH_BINARY)

	Thdd, ContoursCrop, hierarchy1 = cv2.findContours(Segmento, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	for d in ContoursCrop:
			if cv2.contourArea(d) < 1500:
				continue
 
			(x, y, w, h) = cv2.boundingRect(d)
	     
			cv2.rectangle(vc, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.circle(vc,(x+w/2,y+h/2),5,(0,0,255),-1)
			cv2.imshow("Seleccion de Placa", vc)
			cv2.waitKey(0)




vc= cv2.imread("C:\Users\gmsvuser\Documents\PythonExercise\Carro.jpg", 1)
#cv2.imshow("Detector", vc)
###########################################################################Llamada de imagen
vc1 = cv2.cvtColor(vc, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Escala De Grises", vc1)
###########################################################################Imagen en escala de grises
retval, thd = cv2.threshold(vc1, 140, 255, cv2.THRESH_BINARY)
#cv2.imshow("Binarisacion", thd)
######################################################################Binarisacion
Segmentacion(thd)
#CropSegmento(thd)




cv2.waitKey(0)
cv2.destroyAllWindows()