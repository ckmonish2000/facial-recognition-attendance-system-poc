import numpy as np
import face_recognition
import cv2
import os

path = "img"

images =[]
img_name = []

for i in os.listdir(path):
    # getting image
    img = cv2.imread(f'{path}/{i}')
    images.append(img)

    # getting filename
    img_name.append(os.path.splitext(i)[0])


print(img_name)
# function to encode img
def encode_image(images):
    encoded_list = []
    
    for img in images:
        convert_to_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(convert_to_rgb)[0]
        encoded_list.append(encode)
    
    return encoded_list


encode_img = encode_image(images)
print("encoding completed")


web = cv2.VideoCapture(0)

while True:
    success , img = web.read()
    img_resize = cv2.resize(img,(0,0),None,0.25,0.25)
    convert_to_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)

    #finding faces in current frame and encoding them
    faceLoc_current_frame = face_recognition.face_locations(convert_to_rgb)
    encode_current_frame = face_recognition.face_encodings(convert_to_rgb,faceLoc_current_frame)

    print(faceLoc_current_frame)
    # compare image from camera feed
    for encodeFace,faceLoc in zip(encode_current_frame,faceLoc_current_frame):
        compare = face_recognition.compare_faces(encode_img,encodeFace)
        distance = face_recognition.face_distance(encode_img,encodeFace)
        
        minIndex = np.argmin(distance)
        name = img_name[minIndex]

        y1,x2,y2,x1 = faceLoc
        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y1-35),(x2,y2),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

        

    cv2.imshow("webcam",img)
    cv2.waitKey(1)




    

