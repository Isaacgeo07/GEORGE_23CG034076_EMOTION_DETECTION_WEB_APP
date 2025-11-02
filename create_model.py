import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import onnx
from onnxruntime import InferenceSession

# Define a simple emotion detection model
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        # Use a pre-trained ResNet as base
        self.resnet = models.resnet18(pretrained=True)
        # Replace the last layer with our emotion classifier
        self.resnet.fc = nn.Linear(512, 8)  # 8 emotions

    def forward(self, x):
        return F.softmax(self.resnet(x), dim=1)

def create_and_save_model():
    print("Creating a simple emotion detection model...")
    
    # Create model
    model = EmotionCNN()
    model.eval()
    
    # Create dummy input (1 image, 3 channels, 224x224 pixels)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    print("Converting to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        'emotion_detector.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                     'output': {0: 'batch_size'}},
        opset_version=11
    )
    
    # Verify the model
    print("Verifying the ONNX model...")
    onnx_model = onnx.load('emotion_detector.onnx')
    onnx.checker.check_model(onnx_model)
    
    # Test with ONNX Runtime
    print("Testing with ONNX Runtime...")
    ort_session = InferenceSession('emotion_detector.onnx')
    outputs = ort_session.run(
        None,
        {'input': dummy_input.numpy()}
    )
    
    print("Model created and saved successfully!")
    return True

if __name__ == "__main__":
    create_and_save_model()