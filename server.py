from pathlib import Path
from io import BytesIO
import base64
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

app = FastAPI(
    title="Water Pollution YOLO Detection API",
    description="Detect floating waste in uploaded images.",
    version="1.0.0",
)

# Development configuration. Restrict origins before production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO(str(MODEL_PATH))

MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.get("/")
def root():
    return {
        "service": "Water Pollution YOLO Detection API",
        "status": "running",
        "model": MODEL_PATH.name,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model": MODEL_PATH.name,
    }


@app.get("/classes")
def classes():
    names = model.names
    return {
        "classes": [
            {"id": int(class_id), "name": str(name)}
            for class_id, name in names.items()
        ]
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    raw = await file.read(MAX_IMAGE_BYTES + 1)

    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )

    image_array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid supported image.",
        )

    start = time.perf_counter()

    try:
        results = model.predict(
            source=image,
            imgsz=640,
            conf=0.25,
            verbose=False,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(exc)}",
        ) from exc

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    result = results[0]
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_id": class_id,
                "class_name": str(model.names[class_id]),
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                },
            })

    annotated = result.plot()
    success, encoded_image = cv2.imencode(".jpg", annotated)

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Could not encode annotated image.",
        )

    annotated_base64 = base64.b64encode(
        encoded_image.tobytes()
    ).decode("ascii")

    return {
        "success": True,
        "filename": file.filename,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "inference_time_ms": elapsed_ms,
        "detection_count": len(detections),
        "detections": detections,
        "annotated_image_base64": annotated_base64,
    }


@app.post("/predict/annotated")
async def predict_annotated(file: UploadFile = File(...)):
    """Return only the annotated JPEG image."""
    response = await predict(file)

    image_bytes = base64.b64decode(response["annotated_image_base64"])

    return StreamingResponse(
        BytesIO(image_bytes),
        media_type="image/jpeg",
        headers={
            "X-Detection-Count": str(response["detection_count"]),
            "X-Inference-Time-Ms": str(response["inference_time_ms"]),
        },
    )
