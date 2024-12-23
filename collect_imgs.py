
import os
import cv2

# Directory to save the dataset
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes (A-Z) and dataset size per class
number_of_classes = 26  # 26 letters in the alphabet
dataset_size = 100

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop over all classes (A-Z)
for j in range(number_of_classes):
    class_letter = chr(j + 65)  # Convert class index to corresponding letter ('A', 'B', ..., 'Z')
    class_dir = os.path.join(DATA_DIR, class_letter)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Prepare to collect data for class {class_letter}. Press "Q" to begin.')

    # Wait for the user to press 'Q' to start collecting data for the current class
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display instructions
        cv2.putText(frame, f'Class: {class_letter} - Press "Q" to start', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f'Starting data collection for class {class_letter}.')
            break

    # Collect images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the current progress
        cv2.putText(frame, f'Collecting {class_letter}: {counter}/{dataset_size}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Save the frame to the class directory
        resized_frame = cv2.resize(frame, (224, 224))  # Resize for consistency
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), resized_frame)
        counter += 1

        # Delay between captures for better variability
        if cv2.waitKey(100) & 0xFF == ord('e'):  # Press 'E' to exit data collection
            print(f"Exiting data collection for class {class_letter}.")
            break

    print(f'Data collection for class {class_letter} completed.')

    # Provide an option to quit during the loop
    print("Press 'Q' to quit or any other key to continue to the next class.")
    if cv2.waitKey(5000) & 0xFF == ord('q'):
        print("Exiting data collection.")
        break

cap.release()
cv2.destroyAllWindows()
