from flask import Flask, render_template

app = Flask(__name__)

# Camera Test
@app.route('/')
def index():
    return render_template('CameraTest.html')

# Run
if __name__ == '__main__':
    app.run(debug=True)