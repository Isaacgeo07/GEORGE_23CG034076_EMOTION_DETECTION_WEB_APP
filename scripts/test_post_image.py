import requests, base64, time

url_root = 'http://127.0.0.1:8000'
print('Checking server at', url_root)

# wait for server to be ready
for i in range(10):
    try:
        r = requests.get(url_root, timeout=2)
        print('GET / status', r.status_code)
        break
    except Exception as e:
        print('GET attempt', i, 'failed:', e)
        time.sleep(1)
else:
    raise SystemExit('Server did not respond')

# fetch a small public test image (portrait placeholder)
img_url = 'https://upload.wikimedia.org/wikipedia/commons/8/89/Portrait_Placeholder.png'
print('Downloading test image:', img_url)
resp = requests.get(img_url, timeout=10)
resp.raise_for_status()
img_bytes = resp.content
b64 = base64.b64encode(img_bytes).decode('ascii')

body = {'image': 'data:image/png;base64,' + b64, 'type': 'upload'}
print('Posting image to /detect_emotion (this may take a few seconds)')
r2 = requests.post(url_root + '/detect_emotion', json=body, timeout=30)
print('POST status', r2.status_code)
print('Response:', r2.text)
