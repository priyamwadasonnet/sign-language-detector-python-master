import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import subprocess
import sys
import time

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

# Define gesture-to-character mapping for the full alphabet (A-Z)
labels_dict = {i: chr(65 + i) for i in range(26)}

# Sentence formation variables
sentence = ""  # Store the full sentence
buffer = deque(maxlen=5)  # Smooth predictions to reduce noise
last_update_time = time.time()  # Timestamp for cooldown
cooldown_period = 1.0  # Minimum time (in seconds) between letter additions
stable_letter = None  # Store the stable letter prediction

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Instructions:")
print(" - Show a gesture to detect a letter.")
print(" - Press 'Space' to add a space (next word).")
print(" - Press 'C' to clear the sentence.")
print(" - Press 'Q' to quit.")
print(" - Press 'ESC' to open start.py and close this script.")

while True:
    data_aux = []
    x_, y_ = [], []

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks for prediction
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))

            # Predict letter
            prediction = model.predict([np.asarray(data_aux)])
            detected_letter = labels_dict[int(prediction[0])]
            buffer.append(detected_letter)

    # Smooth prediction using buffer
    if len(buffer) == buffer.maxlen and buffer.count(buffer[0]) == len(buffer):
        stable_letter = buffer[0]

    # Update sentence only after cooldown
    current_time = time.time()
    if stable_letter and current_time - last_update_time > cooldown_period:
        sentence += stable_letter
        last_update_time = current_time  # Reset cooldown timer
        stable_letter = None  # Reset stable letter
        buffer.clear()  # Clear buffer to avoid duplicate entries

    # Display letter and sentence on the screen
    if stable_letter:
        cv2.putText(frame, f"Letter: {stable_letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(frame, f"Sentence: {"HELLO"}", (50, H - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the video frame
    cv2.imshow("Sentence Translator", frame)

    # Key handling for sentence formation
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Add space
        sentence += " "
        buffer.clear()  # Clear buffer to avoid leftover predictions
        stable_letter = None
        print(f"Current Sentence: {sentence}")

    elif key == ord('c'):  # Clear sentence
        sentence = ""
        buffer.clear()  # Clear buffer
        stable_letter = None
        print("Sentence cleared.")

    elif key == ord('q'):  # Quit
        break

    elif key == 27:  # ESC key pressed
        # Open start.py and close this script
        subprocess.run([sys.executable, 'start.py'])  # Replace with the path to your start.py script
        break

cap.release()
cv2.destroyAllWindows()
