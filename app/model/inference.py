import io
import time
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple

from ultralytics import YOLO
from PIL import Image

# 🔥 Load model once
model = YOLO("yolov8n.pt")

# 🔥 Config
CONF_THRESHOLD = 0.5
model.conf = CONF_THRESHOLD
model.iou = 0.45

names = model.names


# 🚀 CORE INFERENCE (single source of truth)
def _predict(image: np.ndarray):
    return model(image, imgsz=640, verbose=False)


# 🚀 JSON OUTPUT
def run_inference(image_bytes: bytes) -> Dict[str, Any]:
    start = time.time()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    height, width = image_np.shape[:2]

    results = _predict(image_np)

    detections: List[Dict[str, Any]] = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            cls_id = int(box.cls[0])

            detections.append({
                "class_id": cls_id,
                "class_name": names[cls_id],
                "confidence": round(conf, 3),
                "bbox": [round(x, 2) for x in box.xyxy[0].tolist()]
            })

    latency = round(time.time() - start, 3)

    return {
        "image_size": {"width": width, "height": height},
        "total_objects": len(detections),
        "latency": latency,
        "detections": detections
    }


# 🚀 IMAGE OUTPUT (uses SAME prediction)
def run_inference_image(image_bytes: bytes) -> Tuple[np.ndarray, float]:
    start = time.time()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    results = _predict(image_np)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            cls_id = int(box.cls[0])
            label = f"{names[cls_id]} {conf:.2f}"

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 🎨 Dynamic color per class
            color = (
                (cls_id * 50) % 255,
                (cls_id * 80) % 255,
                (cls_id * 120) % 255
            )

            # Draw bounding box
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)

            # Label background
            (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(image_np, (x1, y1 - h - 10), (x1 + w, y1), color, -1)

            # Label text
            cv2.putText(
                image_np,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    latency = round(time.time() - start, 3)

    return image_np, latency

def annotate_frame(image_np: np.ndarray, results) -> np.ndarray:
    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            cls_id = int(box.cls[0])
            label = f"{names[cls_id]} {conf:.2f}"

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = (
                (cls_id * 50) % 255,
                (cls_id * 80) % 255,
                (cls_id * 120) % 255
            )

            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)

            (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            cv2.rectangle(image_np, (x1, y1 - h - 10), (x1 + w, y1), color, -1)

            cv2.putText(
                image_np,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    return image_np

def generate_video_stream(source=0):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    while True:
        success, frame = cap.read()
        if not success:
            break

        start = time.time()

        # 🔥 Use SAME inference pipeline
        results = _predict(frame)

        # 🔥 Reuse annotation
        annotated = annotate_frame(frame, results)

        latency = time.time() - start

        # Optional: overlay latency
        cv2.putText(
            annotated,
            f"{latency*1000:.1f} ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # 🔥 CRITICAL: Ensure BGR format (OpenCV default is already BGR)
        ret, buffer = cv2.imencode(".jpg", annotated)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )

    cap.release()