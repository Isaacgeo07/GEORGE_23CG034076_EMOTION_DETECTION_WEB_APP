from flask import Flask, render_template, Response, request, jsonify
from emotion_detector import EmotionDetector
import cv2
from waitress import serve
import os

app = Flask(__name__)
emotion_detector = EmotionDetector()  # Load model once (on startup)
camera = cv2.VideoCapture(0)  # Optional webcam

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_emotion():
    # Example: handle uploaded image (instead of live stream)
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    emotion = emotion_detector.predict(frame)
    return jsonify({'emotion': emotion})


# For local testing
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)
