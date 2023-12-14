import cv2 as cv


# ----> Code
#* ---> Heading
#! ---> Important point
#? ---> Things to remember

img = cv.imread('2.jpeg')
# cv.imshow('Persons', img)

gray_person = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray persons', gray_person)

#* Detecting Face using haar_Cascade

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray_person, scaleFactor=1.1, minNeighbors=9)

#! The detectMultiScale function returns a list of rectangles where each rectangle represents a detected face.

#! scaleFactor=1.1: A parameter that compensates for different face sizes. It specifies how much the image size is reduced at each image scale.

#! minNeighbors=3: Another parameter that specifies how many neighbors a candidate rectangle should have to be considered a face. Increasing this value helps reduce false positives.

#! what does false positives mean : 
#? False positives in the context of face detection mean that the algorithm incorrectly identifies regions in an image as faces when, in reality, there are no faces present. In other words, it's a mistake where the algorithm wrongly detects something as a face even though it isn't.

#? faces_rect is essentially a list of rectangles coordinates for the faces that are present in the image, we can looping over this list and essentially grab the coordinates of those images and draw a rectangle over the detected faces

#* Print the detected faces number

print(f'Faces detected are: {len(faces_rect)}') 

#* Print rectangle over the detected faces 

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=1)

cv.imshow('Detected Images', img)

cv.waitKey(0)
cv.destroyAllWindows() 