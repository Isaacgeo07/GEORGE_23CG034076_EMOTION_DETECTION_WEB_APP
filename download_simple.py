import os
import requests
import time

def download_model():
    url = 'https://huggingface.co/spaces/nateraw/emotion-models/resolve/main/emotion-ferplus-8.onnx'
    print(f"Downloading model from {url}...")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        with open('emotion_detector.onnx', 'wb') as f:
            f.write(response.content)
        
        print("Model downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        
        # Try fallback URL
        fallback_url = 'https://huggingface.co/spaces/nateraw/emotion-models/resolve/main/affectnet_emotion.onnx'
        print(f"\nTrying fallback URL: {fallback_url}")
        
        try:
            response = requests.get(fallback_url, headers=headers)
            response.raise_for_status()
            
            with open('emotion_detector.onnx', 'wb') as f:
                f.write(response.content)
            
            print("Model downloaded successfully from fallback URL!")
            return True
        except Exception as e:
            print(f"Error downloading from fallback URL: {e}")
            return False

if __name__ == "__main__":
    download_model()