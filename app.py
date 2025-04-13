from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import numpy as np
import base64
import tensorflow as tf
import mediapipe as mp
import os
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

# Load keras model
try:
    model = tf.keras.models.load_model('model.keras')
    model_loaded = True
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def preprocess_image(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_gesture(frame):
    if not model_loaded:
        return "?", 0
    processed = preprocess_image(frame)
    predictions = model.predict(processed)
    idx = np.argmax(predictions[0])
    confidence = predictions[0][idx]
    letter = chr(ord('a') + idx)
    return letter, float(confidence)

def crop_hand_from_frame(frame):
    # Convert the frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            # Get the bounding box around the hand
            x_min = min([lm.x for lm in landmarks.landmark]) * frame.shape[1]
            x_max = max([lm.x for lm in landmarks.landmark]) * frame.shape[1]
            y_min = min([lm.y for lm in landmarks.landmark]) * frame.shape[0]
            y_max = max([lm.y for lm in landmarks.landmark]) * frame.shape[0]
            # Calculate the center of the hand
            center_x = int((x_min + x_max) / 2)
            center_y = int((y_min + y_max) / 2)
            # Define the zoomed-in region (size 224x224)
            size = 224
            x1 = max(center_x - size // 2, 0)
            y1 = max(center_y - size // 2, 0)
            x2 = min(center_x + size // 2, frame.shape[1])
            y2 = min(center_y + size // 2, frame.shape[0])
            # Extract the region of interest (ROI) around the hand
            cropped_hand = frame[y1:y2, x1:x2]
            return cropped_hand
    # If no hand is detected, return the original frame
    return frame

@app.route('/')
def index():
    return render_template('HomePage.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('frame')
def handle_frame(data):
    try:
        if not data:
            print("Received empty data")
            return

        # Remove base64 header (e.g., data:image/jpeg;base64,...)
        header, encoded = data.split(',', 1)
        decoded = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            print("Failed to decode image")
            return

        # Save captured image to static/captures folder
        captures_dir = os.path.join('static', 'captures')
        if not os.path.exists(captures_dir):
            os.makedirs(captures_dir)
        timestamp = int(time.time())
        filename = os.path.join(captures_dir, f"capture_{timestamp}.jpg")
        cv2.imwrite(filename, frame)

        # Crop the hand region for prediction
        cropped_frame = crop_hand_from_frame(frame)
        # Predict the gesture from the captured (and cropped) frame
        letter, confidence = predict_gesture(cropped_frame)
        # Emit prediction result (letter and confidence)
        socketio.emit('prediction', {
            'letter': letter,
            'confidence': round(confidence * 100, 2)  # as percentage
        })

    except Exception as e:
        print(f"Error processing frame: {e}")

if __name__ == '__main__':
    print("Starting server...")
    socketio.run(app, debug=True, port=8080, allow_unsafe_werkzeug=True)