
from pathlib import Path
from io import BytesIO
import asyncio
import base64
import logging
import os
import time

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from ultralytics import YOLO


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

IMAGE_SIZE = 416
CONFIDENCE = 0.35
MAX_DETECTIONS = 100
JPEG_QUALITY = 85

# Allow one inference at a time per application worker.
# Requests that cannot enter quickly receive HTTP 503.
INFERENCE_QUEUE_TIMEOUT = 0.25
INFERENCE_SEMAPHORE = asyncio.Semaphore(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("water-pollution-yolo")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

# Limit CPU thread oversubscription on small instances.
CPU_THREADS = max(
    1, int(os.getenv("YOLO_CPU_THREADS", "2"))
)

torch.set_num_threads(CPU_THREADS)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # PyTorch may already have initialized its thread pool.
    pass

cv2.setNumThreads(0)


# --------------------------------------------------
# Application and model
# --------------------------------------------------

app = FastAPI(
    title="Water Pollution YOLO Detection API",
    description="Detect floating waste in uploaded images.",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Loading YOLO model: %s", MODEL_PATH)
model = YOLO(str(MODEL_PATH))
logger.info("YOLO model loaded; CPU threads=%s", CPU_THREADS)


# --------------------------------------------------
# Basic endpoints
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "Water Pollution YOLO Detection API",
        "status": "running",
        "model": MODEL_PATH.name,
        "device": "cpu",
        "image_size": IMAGE_SIZE,
        "cpu_threads": CPU_THREADS,
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
    return {
        "classes": [
            {"id": int(class_id), "name": str(name)}
            for class_id, name in model.names.items()
        ]
    }


# --------------------------------------------------
# Synchronous image processing
# Runs in a worker thread so inference does not block
# the FastAPI event loop.
# --------------------------------------------------

def process_image(image: np.ndarray, filename: str):
    height, width = image.shape[:2]

    # Measure inference separately from postprocessing.
    start = time.perf_counter()

    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        max_det=MAX_DETECTIONS,
        device="cpu",
        verbose=False,
    )

    inference_ms = round(
        (time.perf_counter() - start) * 1000, 2
    )

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

    start = time.perf_counter()

    annotated = result.plot()

    success, encoded_image = cv2.imencode(
        ".jpg",
        annotated,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )

    if not success:
        raise RuntimeError("Annotated image encoding failed")

    annotated_bytes = encoded_image.tobytes()

    postprocess_ms = round(
        (time.perf_counter() - start) * 1000, 2
    )

    response_data = {
        "success": True,
        "filename": filename,
        "image_width": int(width),
        "image_height": int(height),
        "inference_time_ms": inference_ms,
        "detection_count": len(detections),
        "detections": detections,
        "timings_ms": {
            "inference": inference_ms,
            "postprocessing_and_encoding": postprocess_ms,
        },
    }

    return response_data, annotated_bytes


# --------------------------------------------------
# Shared upload and inference handling
# --------------------------------------------------

async def run_prediction(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    raw = await file.read(MAX_IMAGE_BYTES + 1)

    if not raw:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )

    decode_start = time.perf_counter()

    image_array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported image.",
        )

    height, width = image.shape[:2]

    if width * height > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail="Image dimensions are too large.",
        )

    decode_ms = round(
        (time.perf_counter() - decode_start) * 1000, 2
    )

    filename = file.filename or "uploaded_image.jpg"

    # Reject overload instead of letting inference requests
    # accumulate and compete for limited CPU/RAM.
    try:
        await asyncio.wait_for(
            INFERENCE_SEMAPHORE.acquire(),
            timeout=INFERENCE_QUEUE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Detection server is busy. Retry shortly.",
            headers={"Retry-After": "2"},
        )

    request_start = time.perf_counter()

    try:
        response_data, annotated_bytes = await asyncio.to_thread(
            process_image, image, filename
        )
    except Exception as exc:
        logger.exception("YOLO prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Check server logs.",
        ) from exc
    finally:
        INFERENCE_SEMAPHORE.release()

    response_data["timings_ms"]["image_decode"] = decode_ms
    response_data["timings_ms"]["server_processing"] = round(
        (time.perf_counter() - request_start) * 1000, 2
    )

    logger.info(
        "prediction filename=%s inference_ms=%s decode_ms=%s "
        "detections=%s",
        filename,
        response_data["inference_time_ms"],
        decode_ms,
        response_data["detection_count"],
    )

    return response_data, annotated_bytes


# --------------------------------------------------
# Prediction endpoints
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        data, annotated_bytes = await run_prediction(file)

        # Preserve the existing API response field.
        data["annotated_image_base64"] = base64.b64encode(
            annotated_bytes
        ).decode("ascii")

        return data
    finally:
        await file.close()


@app.post("/predict/annotated")
async def predict_annotated(file: UploadFile = File(...)):
    try:
        data, annotated_bytes = await run_prediction(file)

        # Return JPEG directly without base64 conversion.
        return Response(
            content=annotated_bytes,
            media_type="image/jpeg",
            headers={
                "X-Detection-Count": str(data["detection_count"]),
                "X-Inference-Time-Ms": str(
                    data["inference_time_ms"]
                ),
                "X-Image-Decode-Ms": str(
                    data["timings_ms"]["image_decode"]
                ),
            },
        )
    finally:
        await file.close()
