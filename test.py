import cv2
import numpy as np
import face_recognition


img1 = face_recognition.load_image_file("./img/register.jpg")
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

img2 = face_recognition.load_image_file("./img/verify.jpg")
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)




cv2.imshow("img 1",img1)
cv2.imshow("img 2",img2)
cv2.waitKey(0)
