from flask import Flask, request, jsonify
import pickle
import numpy as np
import cv2
from mediapipe import solutions as mp
from sklearn.preprocessing import StandardScaler

# Load trained model, label encoder, and scaler
model_data = pickle.load(open('model.p', 'rb'))
model = model_data['model']
label_encoder = model_data['label_encoder']
scaler = model_data['scaler']

# Initialize MediaPipe Hands
mp_hands = mp.hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': f"Error reading image: {str(e)}"}), 400

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return jsonify({'gesture': 'No hand detected'})

    # Process landmarks, scale, and predict gesture
    data_aux = []
    for hand_landmarks in results.multi_hand_landmarks:
        x_, y_ = [], []
        for landmark in hand_landmarks.landmark:
            x_.append(landmark.x)
            y_.append(landmark.y)

        if x_ and y_:
            x_min, y_min = min(x_), min(y_)
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - x_min)
                data_aux.append(landmark.y - y_min)

    # Scale and predict gesture
    data_aux = scaler.transform([data_aux])
    prediction = model.predict(data_aux)
    gesture = label_encoder.inverse_transform(prediction)[0]

    return jsonify({'gesture': gesture})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


