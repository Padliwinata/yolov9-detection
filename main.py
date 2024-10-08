import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from ultralytics import YOLO

class YOLODetectionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 1  # Use default camera
        self.vid = cv2.VideoCapture(self.video_source)

        # Get actual video source width and height
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create canvas with actual video dimensions
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        self.btn_snapshot = tk.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        # Load YOLOv9 model
        self.model = YOLO("yolov9_version_1.pt")
        self.confidence = 0.6
        self.classNames = ["real", "replay", "print"]

        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        ret, frame = self.vid.read()
        if ret:
            cv2.imwrite("snapshot.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            # Process the frame with YOLOv9
            results = self.model(frame, stream=True, verbose=False)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if conf > self.confidence:
                        if self.classNames[cls] == 'real':
                            color = (0, 255, 0)
                        else:
                            color = (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f'{self.classNames[cls].upper()} {int(conf*100)}%', 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Convert the frame to RGB for displaying with tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize the frame to match the canvas size if necessary
            if frame_rgb.shape[1] != self.width or frame_rgb.shape[0] != self.height:
                frame_rgb = cv2.resize(frame_rgb, (self.width, self.height))
            
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
YOLODetectionApp(tk.Tk(), "YOLOv9 Detection")