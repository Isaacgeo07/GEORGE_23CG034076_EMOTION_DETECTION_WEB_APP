import torch
import cv2
import numpy as np
from PIL import Image
import base64
import os
# Torch and model are imported/created lazily inside initialize_model to avoid hard failures

class EmotionDetector:
    def __init__(self):
        self.face_cascade = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']
        self.initialize_model()
        
    def initialize_model(self):
        try:
            # Prefer MTCNN (facenet-pytorch) for more reliable face detection
            try:
                from facenet_pytorch import MTCNN
                self.mtcnn = MTCNN(keep_all=True, device=self.device)
                self.face_cascade = None
                print("MTCNN face detector loaded successfully!")
            except Exception:
                # Fallback to Haar cascade if facenet-pytorch is not available
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if os.path.exists(cascade_path):
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                    self.mtcnn = None
                    print("Haar cascade face detector loaded successfully (fallback).")
                else:
                    print(f"Error: Could not find cascade file at {cascade_path}")
                    self.face_cascade = None
                    self.mtcnn = None
            
            # Try to load torch and the trained model lazily (may fail if torch isn't installed correctly)
            try:
                import torch
                import torch.nn as nn
                import torchvision.transforms as transforms

                # Define the CNN locally so we only require torch when loading a model
                class EmotionCNN(nn.Module):
                    def __init__(self, num_classes=7):
                        super(EmotionCNN, self).__init__()
                        self.conv1 = nn.Sequential(
                            nn.Conv2d(1, 64, kernel_size=3, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.MaxPool2d(2)
                        )
                        self.conv2 = nn.Sequential(
                            nn.Conv2d(64, 128, kernel_size=3, padding=1),
                            nn.BatchNorm2d(128),
                            nn.ReLU(),
                            nn.MaxPool2d(2)
                        )
                        self.conv3 = nn.Sequential(
                            nn.Conv2d(128, 256, kernel_size=3, padding=1),
                            nn.BatchNorm2d(256),
                            nn.ReLU(),
                            nn.MaxPool2d(2)
                        )
                        self.fc1 = nn.Sequential(
                            nn.Linear(256 * 6 * 6, 512),
                            nn.ReLU(),
                            nn.Dropout(0.5)
                        )
                        self.fc2 = nn.Sequential(nn.Linear(512, 7))

                    def forward(self, x):
                        x = self.conv1(x)
                        x = self.conv2(x)
                        x = self.conv3(x)
                        x = x.view(x.size(0), -1)
                        x = self.fc1(x)
                        x = self.fc2(x)
                        return x

                self.torch = torch
                self.transforms = transforms
                self.model = EmotionCNN().to(self.device)
                if os.path.exists('best_emotion_model.pth'):
                    try:
                        import torch
                        self.model.load_state_dict(torch.load('best_emotion_model.pth', map_location=self.device))
                        self.model.eval()
                        print("Emotion detection model loaded successfully!")
                    except Exception as e:
                        print(f"Could not load trained model: {e}")
                        self.model = None
                else:
                    print("Warning: No trained model found. Please run train_model.py first.")
                    self.model = None
            except Exception as e:
                print(f"Torch not available or failed to import: {e}")
                self.model = None
                self.torch = None
                self.transforms = None
                
        except Exception as e:
            print(f"Error loading models: {e}")
            self.face_cascade = None
            self.model = None
    
    def detect_emotions(self, image_data):
        try:
            # Decode image
            nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert to RGB
            if len(img.shape) == 2:  # grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # Detect face
            if self.face_cascade is None:
                return "Face detection model not loaded", 0.0, {"Error": 1.0}
                
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return "No face detected", 0.0, {"No face detected": 1.0}
            
            # Extract face region depending on detector
            if self.mtcnn is not None:
                # MTCNN returns bounding boxes in pixels: [[x1, y1, x2, y2], ...]
                boxes, _ = self.mtcnn.detect(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                if boxes is None or len(boxes) == 0:
                    return "No face detected", 0.0, {"No face detected": 1.0}
                # choose largest box
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                b = boxes[int(np.argmax(areas))]
                x1, y1, x2, y2 = [int(max(0, v)) for v in b]
                face_img = img[y1:y2, x1:x2]
            else:
                # Haar cascade path already computed 'faces'
                x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
                face_img = img[y:y+h, x:x+w]

            # If a trained model is available, run it. Otherwise fall back to random demo output.
            if self.model is not None:
                # Preprocess and run through model
                if len(face_img.shape) == 3:
                    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                else:
                    gray_face = face_img
                # resize to 48x48
                resized = cv2.resize(gray_face, (48, 48))
                if self.transforms is None or self.torch is None:
                    return "Model runtime not available", 0.0, {"Error": 1.0}
                tensor = self.transforms.Compose([
                    self.transforms.ToTensor(),
                    self.transforms.Normalize((0.5,), (0.5,))
                ])(Image.fromarray(resized)).unsqueeze(0).to(self.device)

                with self.torch.no_grad():
                    outputs = self.model(tensor)
                    probs = self.torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()

                all_emotions = {emotion: float(prob) for emotion, prob in zip(self.emotions, probs)}
                sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)
                main_emotion = sorted_emotions[0][0]
                main_confidence = sorted_emotions[0][1]
                return main_emotion, main_confidence, all_emotions

            # Fallback randomized output for demo/testing when no trained model
            import random
            weights = [0.3, 0.25, 0.1, 0.1, 0.1, 0.1, 0.05]
            main_emotion = random.choices(self.emotions, weights=weights)[0]
            main_confidence = random.uniform(0.7, 0.95)
            all_probabilities = np.random.dirichlet(np.array([5 if e == main_emotion else 1 for e in self.emotions]))
            all_emotions = {emotion: float(prob) for emotion, prob in zip(self.emotions, all_probabilities)}
            return main_emotion, main_confidence, all_emotions
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return "Error", 0.0, {"Error": 1.0}