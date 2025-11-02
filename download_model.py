import requests
import sys
import time
import urllib3
import os

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_file(url, filename, verify_ssl=True):
    try:
        print(f"Attempting download from {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, stream=True, timeout=30, headers=headers, verify=verify_ssl)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f:
            if total_size == 0:
                print("Warning: Content length unknown")
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = int(100 * downloaded / total_size)
                        print(f"\rProgress: {percent}%", end='', flush=True)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"Error downloading from {url}: {e}")
        return False

# List of mirrors to try
urls = [
    # Primary HuggingFace mirrors
    "https://huggingface.co/spaces/nateraw/emotion-models/resolve/main/emotion-ferplus-8.onnx",
    "https://huggingface.co/spaces/nateraw/emotion-models/resolve/main/affectnet_emotion.onnx",
    # Backup mirrors
    "https://huggingface.co/akhaliq/emotion-ferplus-8/resolve/main/emotion-ferplus-8.onnx",
    "https://huggingface.co/spaces/microsoft/FERPlus/resolve/main/model.onnx",
    # Community mirrors / archives
    "https://github.com/microsoft/FERPlus/raw/master/models/emotion-ferplus-8.onnx",
    "https://media.githubusercontent.com/media/onnx/models/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
    # AffectNet model fallback (user-provided URL from env)
    os.getenv('EMOTION_MODEL_URL') or ''
]

filename = "emotion_detector.onnx"

# Try each URL until one works
success = False
for url in urls:
    if not url:
        continue
    # Try with SSL verification first
    if download_file(url, filename, verify_ssl=True):
        success = True
        break
    
    # If failed, try without SSL verification
    if download_file(url, filename, verify_ssl=False):
        success = True
        break
    
    time.sleep(1)  # Wait a bit between attempts

if success:
    print(f"Successfully downloaded {filename}")
    sys.exit(0)
else:
    print("Failed to download from all mirrors")
    print("\nPlease try downloading manually from:")
    print("1. https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus")
    print("2. Save the file as 'emotion_detector.onnx' in your project folder")
    sys.exit(1)