import io
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load model once
model = YOLO("yolov8n.pt")

def run_inference(image_bytes: bytes):
    # Convert bytes → image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Run prediction
    results = model(image)

    detections = []

    for r in results:
        for box in r.boxes:
            detections.append({
                "class": int(box.cls[0]),
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })

    return detections