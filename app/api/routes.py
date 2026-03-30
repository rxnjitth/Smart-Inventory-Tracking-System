from fastapi import APIRouter, UploadFile, File, HTTPException
import logging
from app.model.inference import *
from fastapi import Response
import cv2
import numpy as np
import io

from app.model.inference import run_inference_image

router = APIRouter(prefix="/v1")

# 🔥 Proper logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    # ✅ Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    try:
        image_bytes = await file.read()

        # ✅ Run inference
        results = run_inference(image_bytes)

        # ✅ Logging (important for MLOps)
        logger.info(
            f"file={file.filename}, objects={results['total_objects']}, latency={results['latency']}s"
        )

        # ✅ Clean response
        return {
            "filename": file.filename,
            **results
        }

    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Inference failed")




# ✅ Health check (must-have in production)
@router.get("/health")
def health():
    return {"status": "ok"}

from fastapi import Response, HTTPException
import cv2
import numpy as np

@router.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image_np, latency = run_inference_image(image_bytes)

    # ✅ Safety checks
    if image_np is None:
        raise HTTPException(status_code=500, detail="Image processing failed")

    if not isinstance(image_np, np.ndarray):
        raise HTTPException(status_code=500, detail="Invalid image format")

    # 🔥 Convert RGB → BGR (VERY IMPORTANT)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    success, buffer = cv2.imencode(".jpg", image_np)

    if not success:
        raise HTTPException(status_code=500, detail="Image encoding failed")

    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={"X-Latency": str(latency)}
    )

from fastapi import APIRouter
from fastapi.responses import StreamingResponse


@router.get("/detect/video")
def video_stream():
    return StreamingResponse(
        generate_video_stream(0),  # webcam or RTSP URL
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

