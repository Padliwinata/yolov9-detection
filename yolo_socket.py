import base64
from flask import Flask
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import math
import time
import cvzone
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load the YOLOv9 model
model = YOLO("yolov9_version_1.pt")

# Class names for YOLOv9 detections
classNames = ["real", "replay", "print"]

# Confidence and dimension thresholds
confidence = 0.6
min_width_threshold = 250  # Threshold for width (approximate for 20 cm)
min_height_threshold = 300  # Threshold for height (approximate for 20 cm)

prev_frame_time = 0
new_frame_time = 0


@app.route('/')
def index():
    return "WebSocket Image Processing Server Running with YOLOv9!"


@socketio.on('image')
def handle_image(data):
    global prev_frame_time, new_frame_time

    # Decode the base64 image received from the client
    image_data = base64.b64decode(data['image'])
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Check if the image is valid
    if img is None:
        emit('error', {'message': 'Invalid image data'})
        return

    # YOLOv9 processing logic
    new_frame_time = time.time()
    results = model(img, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Confidence score
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class index
            cls = int(box.cls[0])

            # Check if the object meets the size and confidence threshold
            if w > min_width_threshold and h > min_height_threshold and conf > confidence:
                if classNames[cls] == 'real':
                    color = (0, 255, 0)  # Green for 'real'
                else:
                    color = (0, 0, 255)  # Red for other classes

                # Draw bounding box and label
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                                   colorB=color)

    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f'FPS: {fps}')

    # Encode the processed image back to base64
    _, buffer = cv2.imencode('.jpg', img)
    processed_img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Send the processed image back to the client
    emit('processed_image', {'image': processed_img_base64})


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000)
