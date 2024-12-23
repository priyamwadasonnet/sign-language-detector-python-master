import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Data directory containing class folders (A-Z)
DATA_DIR = './data'

# Initialize data and label lists
data = []
labels = []

# Process each class folder (A-Z)
for dir_ in sorted(os.listdir(DATA_DIR)):  # Sort folders alphabetically for consistency
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):  # Ensure it is a folder
        print(f"Processing class: {dir_}")

        for img_file in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_file)
            
            try:
                # Read the image
                img = cv2.imread(img_path)
                if img is None:  # Skip non-readable files
                    print(f"Warning: Could not read {img_path}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Process the image with MediaPipe Hands
                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        data_aux = []
                        x_, y_ = [], []

                        # Extract landmark coordinates
                        for landmark in hand_landmarks.landmark:
                            x_.append(landmark.x)
                            y_.append(landmark.y)

                        # Normalize the landmarks
                        if x_ and y_:
                            x_min, y_min = min(x_), min(y_)
                            for landmark in hand_landmarks.landmark:
                                data_aux.append(landmark.x - x_min)  # Normalize x
                                data_aux.append(landmark.y - y_min)  # Normalize y

                            data.append(data_aux)
                            labels.append(ord(dir_) - 65)  # Convert class label (A-Z) to index (0-25)
                else:
                    print(f"No landmarks detected in {img_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

# Save the data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data collection and saving complete!")
