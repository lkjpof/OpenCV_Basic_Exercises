import numpy as np
import cv2

# Get the necessary harrcascade classifiers (Face and Eyes)
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

# Initialize Camera
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to gray for ease of computation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the Face in a Gray image for ease of computation
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        # Draw the rectange for the face
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Find the eyes in a Gray image for ease of computation
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # Draw the rectangles for the eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # Display the resulting frame
    cv2.imshow('Face and Eyes', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
