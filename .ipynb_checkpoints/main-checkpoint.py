import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Location of the face detection cascade model
FACE_DETECTION = "haarcascade_frontalface_default.xml"
# Location of the face mask detection CNN model
FACE_MASK_DETECTION = "face-mask-detection-model"
# The int or URL corresponding to the capture device
CAPTURE_DEVICE = 0
# List of the masked status
CATEGORIES = ["NO MASK", "MASK"]

# The face detector
faceDetector = cv2.CascadeClassifier(FACE_DETECTION)
# The face mask detector
faceMaskDetector = load_model(FACE_MASK_DETECTION)
# The capturer
cam = cv2.VideoCapture(CAPTURE_DEVICE)

# Function to capture, detect/classify and show the current frame
def engine():
    _, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Get the faces' rects
    faces = faceDetector.detectMultiScale(gray, 1.3, 5)
    
    for x, y, w, h in faces:
        # The window around the face to consider for detecting mask
        window = gray[y : y + h, x : x + w]
        resized = cv2.resize(window, (100, 100))
        reshaped = resized.reshape(100, 100, 1)
        # The probabilities corresponding to the labels
        yprobs = faceMaskDetector.predict(reshaped)
        # Probabilities of the corresponding labels
        p = yprobs[0]
        # The best label predicted
        label = np.argmax(p)
        # The color of the rectangle
        color = (0, int(255 * p[1]), int(255 * p[0]))  # BGR
        # Draw the surrounding rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # Draw the label rectangle (box)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), color, -1)
        # Text denoting the masked status
        cv2.putText(frame, CATEGORIES[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))
        
    # Show the frame
    cv2.imshow("Face Mask Detection", frame)
    
# The loop to run the capture
while cam.isOpened():
    engine()
    if cv2.waitKey(10) & 0xff == ord("q"):
        break
        
cv2.destroyAllWindows()
cam.release()