import os
import cv2 as cv
import numpy as np

main_folder_path = r'C:\Users\attaa\Desktop\Face Recognition System\Faces'

directory_path = os.listdir(main_folder_path)

people_list = [items for items in directory_path]

# for items in os.listdir(main_folder_path):
#     people_list.append(items)


faces = []
names = []
haar_cascade = cv.CascadeClassifier('haar_face.xml')

def model_train():
    for person in people_list:
        #? The Code written below will join folder path to directory_path means:-   C:\Users\attaa\Desktop\Face Recognition System\Faces\Ben Afflek
        path = os.path.join(main_folder_path, person) 
        label = people_list.index(person)

        for img in os.listdir(path):
            #? The Code written below will join folder path to directory_path means:- C:\Users\attaa\Desktop\Face Recognition System\Faces\Ben Afflek\1.jpg
            img_path = os.path.join(path, img)

            img_list = cv.imread(img_path)

            gray = cv.cvtColor(img_list, cv.COLOR_BGR2GRAY)

            #! The detectMultiScale function returns a list of rectangles where each rectangle represents a detected face.

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:

                #? (x, y) are the coordinates of the top-left corner of the rectangle.

                  #? y:y+h and x:x+w are slicing operations on the gray image. They specify the region of interest (ROI) corresponding to the detected face rectangle. 

                  #? 1. y:y+h selects rows from y to y+h. 
                     
                  #? 2. x:x+w selects columns from x to x+w. 

               #? 3.The result, faces_roi, is a cropped region of the grayscale image that contains only the detected face.

                faces_roi = gray[y:y+h, x:x+w]

                faces.append(faces_roi)
                names.append(label)

model_train()

# print(f'Faces are: {len(faces)}')
# print(f'Names are: {len(names)}')

#! How to train the model

print('Training done --------------')

faces = np.array(faces, dtype='object')
names = np.array(names)

face_recoginzer = cv.face.LBPHFaceRecognizer_create()

#! Train modeles on faces and names

face_recoginzer.train(faces, names)

face_recoginzer.save('face_recoginzer.yml')

np.save('Faces.npy', faces)
np.save('Names.npy', names)