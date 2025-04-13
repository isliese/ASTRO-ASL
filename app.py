from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import numpy as np
import base64
import tensorflow as tf
import json

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

# Load model
try:
    model = tf.keras.models.load_model('model.keras')
    model_loaded = True
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

def preprocess_image(frame):
    img = cv2.resize(frame, (200, 200))
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

        # Convert base64 to numpy array
        decoded = base64.b64decode(data)
        nparr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("Failed to decode image")
            return

        # Make prediction
        letter, confidence = predict_gesture(frame)
        
        # Send prediction back to client
        socketio.emit('prediction', {
            'letter': letter,
            'confidence': confidence
        })

        # Debug display
        cv2.imshow('WebSocket Frame', frame)
        if cv2.waitKey(1) == 27:  # ESC
            exit(0)

    except Exception as e:
        print(f"Error processing frame: {e}")

if __name__ == '__main__':
    print("Starting server...")
    socketio.run(app, debug=True, port=8080, allow_unsafe_werkzeug=True)
