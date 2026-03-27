from fastapi import APIRouter, UploadFile, File
from app.model.inference import run_inference

router = APIRouter()

@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    results = run_inference(image_bytes)
    return {"detections": results}