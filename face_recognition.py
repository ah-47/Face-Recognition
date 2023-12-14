import os
import numpy as np 
import cv2 as cv

directory_path = os.listdir(r'C:\Users\attaa\Desktop\Face Recognition System\Faces')

people_list = [items for items in directory_path]

haar_cascade = cv.CascadeClassifier('haar_face.xml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_recoginzer.yml')

faces = np.load('Faces.npy', allow_pickle=True)
names = np.load('Names.npy')


img = cv.imread(r'C:\Users\attaa\Desktop\Face Recognition System\Faces\val\ben_afflek\ben.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Detect the faces in the image

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    names, confidence = face_recognizer.predict(faces_roi)
     
    print(f'Label = {people_list[names]} with a confidence of {confidence}')

    cv.putText(img, str(people_list[names]), (10,30), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,0,255), thickness=2)

    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=1)

cv.imshow('Detected Faces', img)

cv.waitKey(0)