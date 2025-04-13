from flask import Flask, render_template

app = Flask(__name__)

# Home Page
@app.route('/')
def index():
    return render_template('HomePage.html')

# Run
if __name__ == '__main__':
    app.run(debug=True)