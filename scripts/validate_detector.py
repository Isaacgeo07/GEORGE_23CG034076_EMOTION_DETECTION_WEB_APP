"""
Validate the /detect_emotion endpoint using a small set of public images.
Saves a summary and writes failures to failures.csv
"""
import requests, base64, time, csv

URL_ROOT = 'http://127.0.0.1:8000'
HEADERS = {'User-Agent': 'Mozilla/5.0 (validation script)'}

# A curated list of public images (portraits) - some may fail due to remote rate limits.
IMAGE_URLS = [
    'https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=600&h=600&fit=crop',
    'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg',
    'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/1/12/Bill_Gates_2018.jpg',
    'https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=600&h=600&fit=crop',
    'https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=600&h=600&fit=crop',
    'https://images.unsplash.com/photo-1519345182560-3f2917c472ef?w=600&h=600&fit=crop',
    'https://images.unsplash.com/photo-1547425260-76bcadfb4f2c?w=600&h=600&fit=crop',
    'https://images.unsplash.com/photo-1500917293891-ef795e70e1f6?w=600&h=600&fit=crop',
    'https://images.unsplash.com/photo-1527980965255-d3b416303d12?w=600&h=600&fit=crop',
    'https://images.unsplash.com/photo-1517365830460-955ce3ccd263?w=600&h=600&fit=crop',
    'https://images.unsplash.com/photo-1517841905240-472988babdf9?w=600&h=600&fit=crop',
    'https://images.unsplash.com/photo-1527980965255-d3b416303d12?w=600&h=600&fit=crop'
]

SUMMARY = []
FAILURES = []
EMOTION_COUNTER = {}

print('Validating detector against', len(IMAGE_URLS), 'images')

# Ensure server is up
for i in range(8):
    try:
        r = requests.get(URL_ROOT, timeout=3)
        print('Server responded:', r.status_code)
        break
    except Exception as e:
        print('Server not ready (attempt', i, '):', e)
        time.sleep(1)
else:
    raise SystemExit('Server not reachable at ' + URL_ROOT)

for idx, img_url in enumerate(IMAGE_URLS, start=1):
    print(f'[{idx}/{len(IMAGE_URLS)}] Fetching', img_url)
    img_bytes = None
    for attempt in range(3):
        try:
            r = requests.get(img_url, headers=HEADERS, timeout=10)
            r.raise_for_status()
            img_bytes = r.content
            break
        except Exception as e:
            print('  Download attempt', attempt+1, 'failed:', e)
            time.sleep(1)
    if img_bytes is None:
        print('  Failed to download image, skipping')
        FAILURES.append({'url': img_url, 'error': 'download_failed'})
        continue

    b64 = base64.b64encode(img_bytes).decode('ascii')
    body = {'image': 'data:image/jpeg;base64,' + b64, 'type': 'upload'}
    try:
        r = requests.post(URL_ROOT + '/detect_emotion', json=body, timeout=20)
        r.raise_for_status()
        data = r.json()
        main = data.get('emotion')
        conf = data.get('confidence')
        SUMMARY.append({'url': img_url, 'emotion': main, 'confidence': conf, 'raw': data})
        EMOTION_COUNTER[main] = EMOTION_COUNTER.get(main, 0) + 1
        print('  Detected:', main, 'conf:', conf)
    except Exception as e:
        print('  POST/detect failed:', e)
        FAILURES.append({'url': img_url, 'error': str(e)})

# Print summary
print('\nValidation summary:')
print('Total images:', len(IMAGE_URLS))
print('Successful detections:', len(SUMMARY))
print('Failures:', len(FAILURES))
print('Emotion distribution:')
for k, v in EMOTION_COUNTER.items():
    print(' ', k, ':', v)

# Save failures to CSV
if FAILURES:
    with open('scripts/validation_failures.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['url', 'error'])
        w.writeheader()
        for row in FAILURES:
            w.writerow(row)
    print('Wrote failures to scripts/validation_failures.csv')

# Save detailed summary JSON lines
with open('scripts/validation_summary.jsonl', 'w', encoding='utf-8') as f:
    for row in SUMMARY:
        f.write(str(row) + '\n')

print('Done')
