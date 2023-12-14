import os
import numpy as np 
import cv2 as cv

# Get the list of people from the directory
directory_path = r'C:\Users\attaa\Desktop\Face Recognition System\Faces'
people_list = os.listdir(directory_path)

# Load the face cascade and face recognition model
haar_cascade = cv.CascadeClassifier('haar_face.xml')
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_recoginzer.yml')

# Load the face images and labels
faces = np.load('Faces.npy', allow_pickle=True)
names = np.load('Names.npy')

# Open the camera
cap = cv.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

    for (x, y, w, h) in faces_rect:
        # Extract the face region
        faces_roi = gray[y:y+h, x:x+w]

        # Predict the label using the face recognizer
        label, confidence = face_recognizer.predict(faces_roi)
        
        # Print the label and confidence
        print(f'Label = {people_list[label]} with a confidence of {confidence}')

        # Display the name on the frame
        cv.putText(frame, str(people_list[label]), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw a rectangle around the face
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    # Display the frame with rectangles and names
    cv.imshow('Detected Faces', frame)

    # Break the loop if 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv.destroyAllWindows()
