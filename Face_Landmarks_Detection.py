import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# Load the predictor
predictor = dlib.shape_predictor("/home/gun/Documents/DCI-CODE/exercise/Face_Landmarks_Detection/shape_predictor_68_face_landmarks.dat")

# Function to calculate distance between two points
def calculate_distance(points):
    pA = points[0]
    pB = points[1]
    return dist.euclidean(pA, pB)

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

        # Calculate and print the distance between landmarks 30 and 48 (nose tip and right corner of mouth)
        points = [(landmarks.part(30).x, landmarks.part(30).y), (landmarks.part(48).x, landmarks.part(48).y)]
        print("Distance between nose tip and right corner of mouth: ", calculate_distance(points))

    cv2.imshow(winname="Face", mat=frame)

    if cv2.waitKey(delay=1) == 27:  # exit on ESC key
        break

cap.release()
cv2.destroyAllWindows()
