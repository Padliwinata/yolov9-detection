import base64
from flask import Flask
from flask_socketio import SocketIO, emit
import cv2
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return "WebSocket Image Processing Server Running!"

@socketio.on('image')
def handle_image(data):
    # Decode the base64 image received
    image_data = base64.b64decode(data['image'])
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Process the image (create negative)
    negative_img = cv2.bitwise_not(img)

    # Encode the processed image back to base64
    _, buffer = cv2.imencode('.jpg', negative_img)
    negative_img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Send the processed image back to the client
    emit('processed_image', {'image': negative_img_base64})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000)
