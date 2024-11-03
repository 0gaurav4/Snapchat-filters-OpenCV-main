import cv2
import dlib
import numpy as np
from math import hypot

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

# Load the mask image
mask = cv2.imread("mask/mask.png")

# Check if the mask image is loaded successfully
if mask is None:
    print("Error: Mask image could not be loaded. Check the path.")
    exit()

# Initialize the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")

# Main loop to capture video
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame was captured correctly
    if not ret or frame is None:
        print("Failed to capture image")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Process each detected face
    for face in faces:
        # Get face coordinates
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        # Get facial landmarks
        landmarks = predictor(gray, face)
        l = (landmarks.part(2).x, landmarks.part(2).y)  # left corner of the mouth
        r = (landmarks.part(15).x, landmarks.part(15).y)  # right corner of the mouth
        m = (landmarks.part(51).x, landmarks.part(51).y)  # midpoint of the mouth
        face_width = int(hypot(l[0] - r[0], l[1] - r[1]))
        face_height = int(face_width * 0.9)

        # Define the area where the mask will be placed
        top_left = (int(m[0] - face_width / 2), int(m[1] - face_height / 2))

        # Resize the mask to fit the detected face
        face_mask = cv2.resize(mask, (face_width, face_height))
        face_area = frame[top_left[1]: top_left[1] + face_height, top_left[0]: top_left[0] + face_width]

        # Create a mask for the face area
        mask_gray = cv2.cvtColor(face_mask, cv2.COLOR_BGR2GRAY)
        _, face_mask_binary = cv2.threshold(mask_gray, 25, 255, cv2.THRESH_BINARY_INV)

        # Apply the mask to the face area
        face_area_no_face = cv2.bitwise_and(face_area, face_area, mask=face_mask_binary)
        final_mask = cv2.add(face_area_no_face, face_mask)

        # Place the final mask on the frame
        frame[top_left[1]: top_left[1] + face_height, top_left[0]: top_left[0] + face_width] = final_mask

    # Display the resulting frame
    cv2.imshow("Video", frame)

    # Exit loop when 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
