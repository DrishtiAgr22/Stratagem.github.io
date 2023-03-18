import cv2
import face_recognition

# Load the image of the known face
known_image = face_recognition.load_image_file("D:/faceverify/face.jpg")

# Extract the encoding of the known face
known_encoding = face_recognition.face_encodings(known_image)[0]

# Open the camera
cap = cv2.VideoCapture(0)

# Capture a frame
ret, frame = cap.read()

# Convert the frame to RGB
rgb_frame = frame[:, :, ::-1]

# Find all the faces in the RGB frame
face_locations = face_recognition.face_locations(rgb_frame)

# Extract the encoding of the detected face
face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

# Verify if the detected face matches the known face
for face_encoding in face_encodings:
    matches = face_recognition.compare_faces([known_encoding], face_encoding)
    if matches[0]:
        print("Face verified")
    else:
        print("Face not recognized")

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()