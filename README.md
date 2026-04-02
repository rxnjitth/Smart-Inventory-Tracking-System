# Smart Inventory Tracking System

FastAPI service for object detection using a YOLOv8 model. The API accepts an image upload and returns detected objects with class IDs, confidence scores, and bounding boxes.

## Tech Stack

- Python 3.11+
- FastAPI
- Uvicorn
- Ultralytics YOLOv8
- Pillow
- NumPy
- OpenCV (installed for image utilities)

## Project Structure

app/
	main.py              FastAPI app entrypoint
	api/routes.py        API routes
	model/inference.py   YOLO model loading and inference logic
	schemas/response.py  Pydantic response schema (optional usage)
yolov8n.pt             YOLO model weights
pyproject.toml
uv.lock
requirements.txt

## Setup

1. Install uv (if not already installed).

Windows PowerShell:

pip install uv

2. Create the virtual environment and install dependencies from lock/project files.

uv sync

3. Activate the virtual environment (optional, uv run works without manual activation).

Windows PowerShell:

.\.venv\Scripts\Activate.ps1

## Run the API

uv run uvicorn app.main:app --reload

By default, the app runs at:

http://127.0.0.1:8000

## API Endpoints

### GET /

Health/status end point.

Example response:

{
	"message": "YOLOv8 API running"
}

### POST /detect

Accepts an image file as multipart form data and returns detections.

Request field:

- file: image file

PowerShell example:

Invoke-RestMethod -Uri "http://127.0.0.1:8000/detect" -Method Post -Form @{ file = Get-Item ".\sample.jpg" }

Example response:

{
	"detections": [
		{
			"class": 0,
			"confidence": 0.91,
			"bbox": [125.3, 88.1, 402.7, 512.9]
		}
	]
}

## Notes

- Keep yolov8n.pt in the project root unless you update the model path in app/model/inference.py.
- The model is loaded once at startup for better runtime performance.
- Add new dependencies with uv add <package-name> so pyproject.toml and uv.lock stay in sync.
- If you still use requirements.txt for deployment, regenerate it with uv pip compile pyproject.toml -o requirements.txt.
