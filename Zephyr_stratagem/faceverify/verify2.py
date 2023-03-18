import cv2
from cv2 import face
# Load the trained model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('lbph_model.xml')

# Load the test image
test_image = cv2.imread('D:/faceverify/face.jpg')

# Convert the test image to grayscale
gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# Detect faces in the test image
face_cascade = cv2.CascadeClassifier('C:/Users/DRISHTI AGRAWAL/Pythoncodes/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

# Recognize faces in the test image
for (x, y, w, h) in faces:
    # Extract the face ROI
    face_roi = gray_image[y:y+h, x:x+w]
    
    # Recognize the face
    label, confidence = face_recognizer.predict(face_roi)
    
    # Check if the recognized face is the target person with a confidence threshold
    if label == 1 and confidence < 70:
        # Draw a green rectangle around the face
        cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display the recognized label
        cv2.putText(test_image, 'Target Person', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display the test image
cv2.imshow('Test Image', test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
