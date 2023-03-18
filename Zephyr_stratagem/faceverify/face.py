import cv2

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('C:/Users/DRISHTI AGRAWAL/Pythoncodes/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# Open the camera
cap = cv2.VideoCapture(0)

# Capture a frame
ret, frame = cap.read()

# Convert the frame to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale frame
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw a rectangle around each detected face
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the captured frame
cv2.imshow('Face Detection', frame)

# Save the frame as an image file
cv2.imwrite('face.jpg', frame)

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()