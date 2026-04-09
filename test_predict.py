"""Test script for DocAssist predict endpoint"""
import io, http.client, json
from PIL import Image

# Create a test X-ray-like grayscale image (256x256)
img = Image.new('RGB', (256, 256), color=(100, 100, 100))
buf = io.BytesIO()
img.save(buf, format='JPEG')
img_bytes = buf.getvalue()

# Send multipart form POST
boundary = 'TestBoundary12345'
body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
    f'Content-Type: image/jpeg\r\n\r\n'
).encode() + img_bytes + f'\r\n--{boundary}--\r\n'.encode()

try:
    conn = http.client.HTTPConnection('localhost', 8000, timeout=30)
    conn.request('POST', '/api/predict',
        body=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
    )
    resp = conn.getresponse()
    data = resp.read().decode()
    print('Status:', resp.status)
    print('Response:', data)
    if resp.status == 200:
        result = json.loads(data)
        print('\n=== NATIJA ===')
        print('Grade:', result.get('grade'))
        print('Ishonch:', result.get('confidence'), '%')
        print('Tashxis:', result.get('details'))
        print('Manba:', result.get('source'))
    else:
        print('XATO:', data)
except Exception as e:
    print('Connection error:', e)
