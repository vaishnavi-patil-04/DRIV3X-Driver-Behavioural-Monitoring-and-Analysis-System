from ultralytics import YOLO
import cv2
import pandas as pd
from tracker import * 

model = YOLO("yolov8n.pt")

def detect_objects(frame):
    results = model(frame, conf=0.4, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "bbox": box.xyxy[0].tolist(),
                "confidence": float(box.conf)
            })

    return detections