import os
from deepface import DeepFace
import cv2
import time
import numpy as np

# Initialize video capture

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera feed.")
    exit()

# Variables for FPS calculation

prev_time = 0
curr_time = 0

# Customizable settings

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_color = (0, 255, 0)  # Green
box_color = (0, 255, 0)    # Green
thickness = 2
resize_width = 800  # Resize frame width for better display
text_spacing = 20  # Space between text lines

while True:

    # Capture frame-by-frame

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize the frame for better display
    frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * (resize_width / frame.shape[1]))))

    # Calculate FPS

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    try:
        # Convert the frame to RGB (DeepFace expects RGB images)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Analyze the frame for age, gender, and emotion

        results = DeepFace.analyze(rgb_frame, actions=["age", "gender", "emotion"], enforce_detection=False)

        # Loop through all detected faces (if multiple faces are present)

        for result in results:
            # Extract results
            age = result["age"]
            gender = result["dominant_gender"]
            emotion = result["dominant_emotion"]
            face_confidence = result["face_confidence"]
            face_region = result["region"]

            # Create an age range (e.g., ±3 years around the predicted age)

            age_range = f"{max(age - 3, 0)} - {age + 3}"

            # Extract face region coordinates

            x, y, w, h = face_region["x"], face_region["y"], face_region["w"], face_region["h"]

            # Draw bounding box around the face

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, thickness)

            # Prepare text to display

            text_age = f"Age: {age_range}"
            text_gender = f"Gender: {gender}"
            text_emotion = f"Emotion: {emotion}"
            text_confidence = f"Confidence: {face_confidence:.2f}"

            # Calculate text position above the bounding box
            text_x = x
            text_y = y - 10  # Start 10 pixels above the bounding box

            # Display text with even spacing
            
            cv2.putText(frame, text_age, (text_x, text_y), font, font_scale, font_color, thickness)
            cv2.putText(frame, text_gender, (text_x, text_y - text_spacing), font, font_scale, font_color, thickness)
            cv2.putText(frame, text_emotion, (text_x, text_y - 2 * text_spacing), font, font_scale, font_color, thickness)
            cv2.putText(frame, text_confidence, (text_x, text_y - 3 * text_spacing), font, font_scale, font_color, thickness)

    except Exception as e:
        print(f"Error during face analysis: {e}")

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), font, font_scale, font_color, thickness)

    # Display the frame
    cv2.imshow("Real-Time Face Analysis (Mobile Camera)", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera feed and close windows
cap.release()
cv2.destroyAllWindows()