# Mood Ring — Emotion Detection Web App

This repository contains a full-stack Flask web app called "Mood Ring" that detects human emotions from an uploaded image or live camera capture.

Features
- Sign up / Sign in
- Name input to personalize feedback
- Image upload or live capture (camera)
- Emotion detection using an ONNX model (AffectNet/FERPlus compatible)
- Feedback with recommendations and reflective questions
- Results are saved to a local SQLite database

Files and structure
- `app.py` — Flask backend, routing, database access, and ONNX inference
- `model.py` — Model training/finetuning script (Keras/TF -> ONNX conversion)
- `templates/index.html` — Single-page frontend (all screens in one file)
- `static/style.css` — Single CSS file for the UI
- `Requirements.txt` — Python packages required
- `Link-to Web-App.txt` — Guidance for deployment and hosting
- `mood.env` — Environment variables (EMOTION_MODEL_URL, FLASK_SECRET_KEY)
- `mood_ring.db` — SQLite database (created on first run)
- `emotion_detector.onnx` — The ONNX model (place manually or set EMOTION_MODEL_URL)

Quick start (development)
1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r Requirements.txt
```

3. If you don't have an ONNX model yet, download one and place it in the repo root as `emotion_detector.onnx`, or set the `EMOTION_MODEL_URL` in `mood.env`.

4. Run the app:

```powershell
python app.py
```

5. Open http://127.0.0.1:5000 in your browser.

Notes & troubleshooting
- If automatic model download fails, manually download a compatible ONNX model and save it as `emotion_detector.onnx` in the project root.
- Training a robust emotion model requires a labeled dataset (e.g., AffectNet, FER2013). `model.py` includes a training pipeline and conversion to ONNX but training is time-consuming and may require a GPU.

Training guidance
- Prepare your dataset in the required folder structure (folders per emotion)
- Run `python model.py` to start training and produce an ONNX model

If you'd like, I can:
- Attempt to download a different pre-trained ONNX model automatically
- Add a small script to initialize the database and create an admin user
- Wire up deployment settings (Procfile/Gunicorn) for Render/Heroku

Tell me what you'd like me to do next and I'll continue implementing it.