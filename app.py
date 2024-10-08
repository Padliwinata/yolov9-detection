from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import cvzone
import math
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI()

# Initialize YOLO model
model = YOLO("yolov9_version_1.pt")  # Load your YOLOv9 model
confidence = 0.6
min_width_threshold = 250  # Threshold for width (approximate for 20 cm)
min_height_threshold = 300  # Threshold for height (approximate for 20 cm)

# Function to process image frames
def process_frame(frame: bytes):
    try:
        # Convert the byte array to a numpy array and then to a format usable by OpenCV
        img_array = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Decode to image format

        if img is None:
            raise ValueError("Decoding image failed; check the input format.")

        results = model(img, stream=True, verbose=False)  # Perform inference

        classNames = ["real", "replay", "print"]  # Define your class names

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])

                # Check if the detected object meets size and confidence thresholds
                if w > min_width_threshold and h > min_height_threshold and conf > confidence:
                    if classNames[cls] == 'real':
                        color = (0, 255, 0)  # Green for 'real'
                    else:
                        color = (0, 0, 255)  # Red for other classes

                    # Draw bounding box and label on the image
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                    cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                       (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                                       colorB=color)

        # Convert back to PIL image for returning
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        return Image.fromarray(img_rgb)

    except Exception as e:
        print(f"Error processing frame: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Define an API endpoint to accept frames
@app.post("/process-frame/")
async def process_frame_endpoint(file: UploadFile = File(...)):
    # Read the file content (image frame) in binary mode
    content = await file.read()  # No need for decode since we're working with bytes

    # Check if the uploaded file is an image
    if not content:
        raise HTTPException(status_code=400, detail="No content uploaded.")
    
    processed_img = process_frame(content)  # Process the frame

    # Return the processed frame back as an image
    buffer = BytesIO()
    processed_img.save(buffer, format="JPEG")
    buffer.seek(0)  # Move the cursor to the beginning of the buffer
    return buffer.getvalue()

# Run the app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

