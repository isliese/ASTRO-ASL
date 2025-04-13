from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

# Camera Test Page
@app.route('/')
def index():
    return render_template('CameraTest.html')

@socketio.on('frame')
def handle_frame(data):
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is not None:
        cv2.imshow('WebSocket Frame', frame)
        if cv2.waitKey(1) == 27:  # ESC
            exit(0)

# Run
if __name__ == '__main__':
    app.run(debug=True,port=8080)