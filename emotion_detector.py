import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import gc

class EmotionDetector:
    def __init__(self):
        # Force CPU only — Render free tier has no GPU and only ~512 MB RAM
        self.device = torch.device('cpu')

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'model', 'emotion_model_optimized.pt')

        # Load lightweight TorchScript model
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        # Small transform to minimize processing
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ])

        # Haarcascade (tiny XML face detector)
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(base_dir, 'haarcascade_frontalface_default.xml')
        )

        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def predict(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return "No face detected"

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            img = Image.fromarray(roi).convert('L')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(img_tensor)
                pred = torch.argmax(output, dim=1).item()

            # Release memory
            del img_tensor, output
            gc.collect()
            torch.cuda.empty_cache()

            return self.emotions[pred]
