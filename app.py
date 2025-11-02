from flask import Flask, render_template, request, jsonify
from emotion_detector import EmotionDetector
from waitress import serve
import numpy as np
import cv2
import os

app = Flask(__name__)
emotion_detector = EmotionDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    emotion = emotion_detector.predict(frame)
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    # 👇 Render provides PORT in the environment — must use it exactly
    port = int(os.environ.get('PORT', 10000))
    print(f"✅ Starting server on port {port}...")
    serve(app, host='0.0.0.0', port=port)
