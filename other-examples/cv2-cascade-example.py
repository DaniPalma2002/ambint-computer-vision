import time
import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# To capture video from webcam. 
# Use the correct index like 0 or -1 for your laptop camera; it might differ based on your setup
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    #_, img = cap.read()
    # If you want to use a image
    img = cv2.imread('3.jpg')

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.05, 1, minSize=(10, 10))
    profiles = profile_cascade.detectMultiScale(gray, 1.1, 4, minSize=(10, 10))

    # Draw the rectangle around each face
    i = 0
    for (x, y, w, h) in faces:
        i += 1
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    for (x, y, w, h) in profiles:
        i += 1
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the number of faces
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Faces ' + str(i), (50, 100), font, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()