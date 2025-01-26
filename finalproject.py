#importing libraries packages to be used in this project
import face_recognition
import cv2
import os
import numpy as np
import csv
from datetime import datetime

# To capture video from webcam.
video_capture = cv2.VideoCapture(0)

#LOADING IMAGES
#importing all the images and encoding the face in the image
modiji_image = face_recognition.load_image_file("C:/Users/omen/Documents/project/Resource/1.jpg")
modiji_encodings = face_recognition.face_encodings(modiji_image)[0]

messi_image = face_recognition.load_image_file("C:/Users/omen/Documents/project/Resource/2.jpg")
messi_encodings = face_recognition.face_encodings(messi_image)[0]

ronaldo_image = face_recognition.load_image_file("C:/Users/omen/Documents/project/Resource/3.jpg")
ronaldo_encodings = face_recognition.face_encodings(ronaldo_image)[0]

#INITIALISING
known_face_encodings = [modiji_encodings, messi_encodings,ronaldo_encodings]
known_face_names = ["Narendra Modi", "Lionel Messi","Cristiano Ronaldo"]

#Creating a copy of known face images
students = known_face_names.copy()

face_locations = []
face_encodings = []

s = True

#Importing Date and time
now = datetime.now()
current_date = now.strftime("%Y/%m/%d")

#Creating a directory if does not exist 
os.makedirs(current_date, exist_ok=True)

#Opening a Csv file
f = open(current_date + '.csv', 'a+', newline='')

#Creating a writer object for the csv file
csv_writer = csv.writer(f)
csv_writer.writerow(["Name", "Time"])

#Main loop for Face Recognition
while True:
    # Read the frame
    _, frame = video_capture.read()
    # Converting to RGB format
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if s:
        #getting the face locations in a frame and encoding the face
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        for face_encoding in face_encodings:
            name=""
            #Comparing the face encodings with the known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            #Finding the best match
            best_match_index = np.argmin(face_distance)
            #If match is found, marking the attendance
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            if name in known_face_names:  
                if name in students:
                    students.remove(name)
                    print(name)
                    current_time = now.strftime("%H:%M:%S")
                    csv_writer.writerow([name, current_time])

    cv2.imshow("attendance system", frame)
    #Exit Condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release resources and closing the csv file
video_capture.release()
cv2.destroyAllWindows()
f.close()
