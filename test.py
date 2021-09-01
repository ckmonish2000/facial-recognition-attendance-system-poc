import cv2
import numpy as np
import face_recognition

# load image and convert from bgr to rgb
img1 = face_recognition.load_image_file("./img/register.jpg")
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

img2 = face_recognition.load_image_file("./img/verify.jpg")
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)


# find face loc and convert img to convolution
faceLoc = face_recognition.face_locations(img1)[0]
encode1 = face_recognition.face_encodings(img1)[0]
cv2.rectangle(img1,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLoc2 = face_recognition.face_locations(img2)[0]
encode2 = face_recognition.face_encodings(img2)[0]
cv2.rectangle(img2,(faceLoc2[3],faceLoc2[0]),(faceLoc2[1],faceLoc2[2]),(255,0,255),2)


result = face_recognition.compare_faces([encode1],encode2)
face_distance = face_recognition.face_distance([encode1],encode2)
cv2.putText(img2,f'{result} {round(face_distance[0],2)}',(90,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow("img 1",img1)
cv2.imshow("img 2",img2)
cv2.waitKey(0)

