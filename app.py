from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        # start capturing frames
        success, frame = camera.read()
        if not success:
            break
        else:
            # converts opencv image to jpeg for html page
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # stream frames as live video
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('CameraTest.html')

@app.route('/video_feed')
def video_feed():
    # streams frames to buffer
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
