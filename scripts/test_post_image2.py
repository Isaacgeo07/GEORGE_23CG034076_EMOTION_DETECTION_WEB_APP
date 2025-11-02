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

candidates = [
    'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg',
    'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/1/12/Bill_Gates_2018.jpg',
    'https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=600&h=600&fit=crop'
]
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'}
img_bytes = None
used = None
for u in candidates:
    try:
        print('Trying', u)
        r = requests.get(u, timeout=10, headers=headers)
        r.raise_for_status()
        img_bytes = r.content
        used = u
        print('Downloaded from', u, 'size', len(img_bytes))
        break
    except Exception as e:
        print('Failed to download from', u, 'error:', e)

if img_bytes is None:
    raise SystemExit('Could not download any candidate images')

b64 = base64.b64encode(img_bytes).decode('ascii')
body = {'image': 'data:image/jpeg;base64,' + b64, 'type': 'upload'}
print('Posting image from', used)
r2 = requests.post(url_root + '/detect_emotion', json=body, timeout=30)
print('POST status', r2.status_code)
print('Response:', r2.text)
