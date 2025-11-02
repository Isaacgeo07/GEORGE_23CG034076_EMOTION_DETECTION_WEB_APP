import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import os
import requests
from tqdm import tqdm
import pandas as pd

# Define the CNN architecture
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        # Convolutional layers
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
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class FERDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_data = self.data.iloc[idx, 1]
        img_array = np.fromstring(img_data, dtype=int, sep=' ').reshape(48, 48)
        image = Image.fromarray(img_array.astype('uint8'))
        
        if self.transform:
            image = self.transform(image)
            
        label = self.data.iloc[idx, 0]
        return image, label


class SimpleArrayDataset(Dataset):
    """Small helper dataset when using synthetic numpy arrays"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx].astype('uint8'))
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[idx])
        return img, label

def download_dataset():
    # Download FER2013 dataset
    dataset_url = "https://www.dropbox.com/s/l6dp4n2r8i3x065/fer2013.csv?dl=1"
    if not os.path.exists('fer2013.csv'):
        print("Downloading FER2013 dataset...")
        response = requests.get(dataset_url, stream=True)
        with open('fer2013.csv', 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                if chunk:
                    f.write(chunk)
    
    print("Dataset downloaded!")

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    best_valid_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        valid_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc='Validation'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        valid_loss = valid_loss / len(valid_loader)
        valid_acc = 100. * correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2f}%')
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_emotion_model.pth')
            print("Saved best model!")
        
        print('-' * 50)

def main():
    parser = argparse.ArgumentParser(description='Train emotion CNN (FER2013)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--quick', action='store_true', help='Quick mode: use small samples and 1 epoch')
    parser.add_argument('--train-sample', type=int, default=2000, help='Number of training samples to use in quick mode')
    parser.add_argument('--valid-sample', type=int, default=500, help='Number of validation samples to use in quick mode')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # If quick mode is requested, create a small synthetic dataset to validate the training pipeline
    if args.quick:
        print('Quick mode enabled: creating synthetic dataset for fast test')
        train_n = args.train_sample
        valid_n = args.valid_sample
        epochs = 1
        batch_size = min(args.batch_size, 32)

        # Synthetic grayscale images (48x48) and random labels (0-6)
        train_images = np.random.randint(0, 256, size=(train_n, 48, 48), dtype=np.uint8)
        train_labels = np.random.randint(0, 7, size=(train_n,))
        valid_images = np.random.randint(0, 256, size=(valid_n, 48, 48), dtype=np.uint8)
        valid_labels = np.random.randint(0, 7, size=(valid_n,))

        transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = SimpleArrayDataset(train_images, train_labels, transform=transform)
        valid_dataset = SimpleArrayDataset(valid_images, valid_labels, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        # Download dataset
        download_dataset()

        # Load and preprocess data
        print("Loading dataset...")
        df = pd.read_csv('fer2013.csv')
        train_data = df[df['Usage'] == 'Training'].reset_index(drop=True)
        valid_data = df[df['Usage'] == 'PublicTest'].reset_index(drop=True)

        # If not quick, use full parameters
        epochs = args.epochs
        batch_size = args.batch_size

        transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Create datasets
        train_dataset = FERDataset(train_data, transform=transform)
        valid_dataset = FERDataset(valid_data, transform=transform)

        # Create data loaders (use num_workers=0 for Windows compatibility)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model, loss function, and optimizer
    model = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Starting training...")
    train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=epochs, device=device)
    print("Training completed!")

if __name__ == "__main__":
    main()