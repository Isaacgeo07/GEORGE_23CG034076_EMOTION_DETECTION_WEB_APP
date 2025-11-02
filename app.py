from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import cv2
import numpy as np
import base64
import os
from dotenv import load_dotenv
from emotion_detector import EmotionDetector

# Load environment variables
load_dotenv('mood.env')

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Database initialization
def init_db():
    conn = sqlite3.connect('mood_ring.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Emotion detections table
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  name TEXT NOT NULL,
                  image_data BLOB,
                  emotion TEXT NOT NULL,
                  confidence REAL,
                  detection_type TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

init_db()

# Initialize emotion detector on startup
emotion_detector = EmotionDetector()

# AffectNet emotion labels (8 emotions)
EMOTION_LABELS = ['Neutral', 'Happy', 'Sad', 'Surprised', 'Fear', 'Disgust', 'Anger', 'Contempt']

# Emotion recommendations
EMOTION_RECOMMENDATIONS = {
    'Happy': {
        'message': 'Your radiant energy is contagious! Keep spreading those positive vibes.',
        'recommendations': [
            'Share your happiness with loved ones - call a friend or family member',
            'Document this moment in a journal to revisit later',
            'Channel this energy into a creative project or hobby',
            'Practice gratitude by listing three things that made you smile today'
        ],
        'questions': [
            'What sparked this beautiful mood today?',
            'Who would you like to share this joy with?',
            'How can you preserve this feeling for challenging days ahead?'
        ]
    },
    'Sad': {
        'message': 'It\'s okay to feel blue sometimes. Your feelings are valid, and brighter days are ahead.',
        'recommendations': [
            'Reach out to someone you trust - connection can help ease the weight',
            'Engage in gentle self-care: a warm bath, your favorite comfort food, or soft music',
            'Write down your thoughts in a journal to process your emotions',
            'Take a walk in nature or get some fresh air to shift your perspective'
        ],
        'questions': [
            'What would comfort you most right now?',
            'Is there someone who could support you through this?',
            'What small step could you take today to nurture yourself?'
        ]
    },
    'Angry': {
        'message': 'Your anger is telling you something important. Let\'s channel that energy constructively.',
        'recommendations': [
            'Try physical exercise to release the tension - even a quick walk helps',
            'Practice deep breathing: inhale for 4, hold for 4, exhale for 6',
            'Write an unsent letter expressing your feelings freely',
            'Identify the root cause - what boundary was crossed or need was unmet?'
        ],
        'questions': [
            'What triggered this emotion?',
            'What do you need to feel heard and understood?',
            'How can you address this situation when you feel calmer?'
        ]
    },
    'Surprised': {
        'message': 'Life just threw you a curveball! Embrace the unexpected.',
        'recommendations': [
            'Take a moment to process what just happened',
            'Share the surprise with someone who would appreciate it',
            'Reflect on how this changes your perspective or plans',
            'Stay open to the opportunities this surprise might bring'
        ],
        'questions': [
            'Was this a pleasant surprise or an unwelcome shock?',
            'How are you adapting to this new information?',
            'What does this surprise reveal about your expectations?'
        ]
    },
    'Fear': {
        'message': 'Feeling afraid shows you\'re human and aware. Let\'s find your courage together.',
        'recommendations': [
            'Ground yourself in the present: name 5 things you can see, 4 you can touch, 3 you can hear',
            'Talk to someone you trust about what\'s worrying you',
            'Break down the fear into smaller, manageable pieces',
            'Remind yourself of past challenges you\'ve overcome'
        ],
        'questions': [
            'What specifically are you afraid of?',
            'What\'s the worst that could happen, and how likely is it?',
            'What resources or support do you have to face this?'
        ]
    },
    'Disgust': {
        'message': 'Your boundaries are speaking up. This feeling helps protect your values.',
        'recommendations': [
            'Remove yourself from the source of disgust if possible',
            'Cleanse your palate or environment - fresh air, clean space, pleasant scents',
            'Reflect on what values or boundaries this reaction is protecting',
            'Channel this into positive action if something needs to change'
        ],
        'questions': [
            'What specifically triggered this reaction?',
            'Is this protecting you from something harmful?',
            'What action, if any, should you take in response?'
        ]
    },
    'Neutral': {
        'message': 'You\'re in a balanced state - a perfect foundation for whatever comes next.',
        'recommendations': [
            'Use this calm moment for planning or reflection',
            'Practice mindfulness to stay present and grounded',
            'Check in with yourself - are there any underlying emotions to explore?',
            'This is a great time for productive work or creative thinking'
        ],
        'questions': [
            'What would you like to focus your energy on right now?',
            'Are you content in this state or seeking something more?',
            'How can you make the most of this balanced moment?'
        ]
    },
    'Contempt': {
        'message': 'You\'re feeling superior or dismissive. Let\'s examine this reaction with compassion.',
        'recommendations': [
            'Pause and consider the other perspective with empathy',
            'Reflect on whether this judgment is serving you well',
            'Channel this into constructive feedback rather than criticism',
            'Consider if underlying hurt or insecurity is fueling this feeling'
        ],
        'questions': [
            'What standard or value feels violated here?',
            'Could there be circumstances you\'re not aware of?',
            'How can you address this situation with more kindness?'
        ]
    }
}



from emotion_detector import EmotionDetector

# Initialize the emotion detector globally
emotion_detector = EmotionDetector()

def detect_emotion(image_data):
    """Detect emotion from image data using EmotionDetector"""
    global emotion_detector
    
    try:
        # Decode image
        nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process image and detect emotions
        main_emotion, main_confidence, all_emotions = emotion_detector.detect_emotions(image_data)
        return main_emotion, main_confidence, all_emotions
        
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return "Error", 0.0, {"Error": 1.0}

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        conn = sqlite3.connect('mood_ring.db')
        c = conn.cursor()
        c.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['username'] = username
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials'})
    
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password required'})
    
    password_hash = generate_password_hash(password)
    
    try:
        conn = sqlite3.connect('mood_ring.db')
        c = conn.cursor()
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                  (username, password_hash))
        conn.commit()
        user_id = c.lastrowid
        conn.close()
        
        session['user_id'] = user_id
        session['username'] = username
        return jsonify({'success': True})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'message': 'Username already exists'})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/save_name', methods=['POST'])
def save_name():
    data = request.json
    session['current_name'] = data.get('name')
    return jsonify({'success': True})

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion_route():
    data = request.json
    image_data = data.get('image')
    detection_type = data.get('type', 'upload')
    
    emotion, confidence, all_emotions = detect_emotion(image_data)
    
    # Save to database
    conn = sqlite3.connect('mood_ring.db')
    c = conn.cursor()
    
    # Convert base64 to blob for storage
    image_blob = base64.b64decode(image_data.split(',')[1])
    
    c.execute('''INSERT INTO detections 
                 (user_id, name, image_data, emotion, confidence, detection_type)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (session.get('user_id'), session.get('current_name', 'Anonymous'),
               image_blob, emotion, confidence, detection_type))
    conn.commit()
    conn.close()
    
    # Get recommendations for the primary emotion
    recommendations = EMOTION_RECOMMENDATIONS.get(emotion, EMOTION_RECOMMENDATIONS['Neutral'])
    
    # Get recommendations for secondary emotions
    secondary_recommendations = []
    for emotion_name, confidence in list(all_emotions.items())[1:]:  # Skip the primary emotion
        if emotion_name.capitalize() in EMOTION_RECOMMENDATIONS:
            rec = EMOTION_RECOMMENDATIONS[emotion_name.capitalize()]
            secondary_recommendations.append({
                'emotion': emotion_name.capitalize(),
                'confidence': confidence,
                'message': rec['message'],
                'recommendations': rec['recommendations'][:2]  # Only include top 2 recommendations
            })
    
    return jsonify({
        'emotion': emotion,
        'confidence': confidence,
        'name': session.get('current_name', 'Friend'),
        'recommendations': recommendations,
        'all_emotions': all_emotions,
        'secondary_recommendations': secondary_recommendations
    })


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8000)