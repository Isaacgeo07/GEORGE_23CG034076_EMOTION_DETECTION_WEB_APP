import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import gc


class EmotionDetector:
    def __init__(self):
        # Force CPU mode for Render free tier
        self.device = torch.device('cpu')

        # Automatically find model path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'model', 'emotion_model_optimized.pt')

        # Load TorchScript (optimized) model if available, else fallback
        if os.path.exists(model_path):
            self.model = torch.jit.load(model_path, map_location=self.device)
        else:
            # fallback: load the normal model (ensure it’s small)
            model_path = os.path.join(base_dir, 'model', 'emotion_model.pth')
            self.model = torch.load(model_path, map_location=self.device)

        self.model.eval()

        # Lightweight transform (resize early)
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ])

        # Load face detector (lightweight XML file)
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(base_dir, 'haarcascade_frontalface_default.xml')
        )

    def predict(self, frame):
        # Convert and preprocess frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return "No face detected"

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (48, 48))
            image = Image.fromarray(roi).convert("L")

            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            output = self.model(img_tensor)
            prediction = torch.argmax(output, dim=1).item()

            # Replace with your own emotion labels
            emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            result = emotions[prediction]

            # Free memory after prediction
            del img_tensor, output
            gc.collect()
            torch.cuda.empty_cache()

            return result
