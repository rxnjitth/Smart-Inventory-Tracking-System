from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="YOLOv8 MLOps API",
    version="1.0.0"
)

# ✅ CORS (keep this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Include API routes
app.include_router(router)

# ✅ Serve static files (important)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 🚀 Root → UI
@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")