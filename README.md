
# YOLO11 Floating Waste Detection API

An AI-powered floating waste detection backend built using **Ultralytics YOLO11 and FastAPI** for image-based waste detection in water pollution monitoring applications.

## Features

- YOLO11-based floating waste detection
- FastAPI REST API
- Nine supported waste categories
- Confidence scores and bounding-box coordinates
- Annotated image generation
- Interactive API documentation with Swagger UI
- Integration-ready for water pollution monitoring dashboards

## Supported Classes

1. Aluminium can
2. Carton
3. Paper
4. Plastic
5. Plastic bag
6. Plastic bottle
7. Plastic container
8. Plastic cup
9. Plastic packaging

## Technology Stack

- Python
- Ultralytics YOLO11
- FastAPI
- Uvicorn
- OpenCV
- NumPy

## Project Structure

```text
YOLO11/
├── server.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/
    └── best.pt
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Jishnu-ms/YOLO11.git
cd YOLO11
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the API Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

## API Documentation

Once the server is running, access the following URLs:

- **API Home:** http://127.0.0.1:8000/
- **Health Check:** http://127.0.0.1:8000/health
- **Supported Classes:** http://127.0.0.1:8000/classes
- **Interactive Swagger UI:** http://127.0.0.1:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information and status |
| GET | `/health` | Health check and model status |
| GET | `/classes` | Returns supported waste classes |
| POST | `/predict` | Detects waste in an uploaded image and returns JSON results |
| POST | `/predict/annotated` | Returns an annotated JPEG image |

### Image Prediction

The `/predict` endpoint accepts an uploaded image in JPEG, PNG, or WebP format.

The response contains:

- Detected object classes
- Confidence scores
- Bounding-box pixel coordinates
- Image dimensions
- Inference time
- Annotated image encoded in Base64

### Example Prediction Response

```json
{
  "success": true,
  "filename": "river_sample.jpg",
  "image_width": 1280,
  "image_height": 720,
  "inference_time_ms": 254.28,
  "detection_count": 1,
  "detections": [
    {
      "class_id": 5,
      "class_name": "plastic bottle",
      "confidence": 0.87,
      "bbox": {
        "x1": 220.5,
        "y1": 140.2,
        "x2": 310.7,
        "y2": 295.4
      }
    }
  ],
  "annotated_image_base64": "..."
}
```

*The example values above are illustrative. Actual detections depend on the uploaded image.*

## Model Training and Evaluation

The YOLO11 model was fine-tuned using a floating-waste dataset containing nine object categories. The dataset was prepared with a grouped train-validation-test split to reduce potential leakage from visually similar images across splits.

### Held-Out Test Results

| Metric | Score |
|---|---:|
| Precision | 90.78% |
| Recall | 85.20% |
| mAP@50 | 91.00% |
| mAP@50–95 | 63.29% |

These metrics represent evaluation on the prepared held-out test dataset. Real-world performance may vary with lighting, water reflections, camera distance, object size, and environmental conditions.

## Water Pollution Monitoring Integration

This API provides the visual waste detection component of an AI-enabled water pollution monitoring system.

Potential integrations include:

- Camera images captured by a monitoring boat
- GPS coordinates collected by a separate navigation system
- Water-quality measurements from pH, turbidity, and TDS sensors
- A web dashboard for visualizing detections and pollution observations
- Pollution heat-map generation using geotagged observations

**Important:** YOLO11 requires only an image for object detection. GPS coordinates and water-quality measurements are separate inputs and are not required by the `/predict` endpoint.

The bounding boxes returned by YOLO11 represent pixel positions within an image, not geographical coordinates. Associating detections with locations requires GPS data and appropriate timestamp matching.

## Deployment

The API can be deployed as a Python web service on a compatible hosting platform such as Render.

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
uvicorn server:app --host 0.0.0.0 --port $PORT
```

Ensure that the deployed service has access to `models/best.pt` at the path expected by `server.py`.

Before deployment, verify that the hosting environment has sufficient memory, disk space, and build capacity for PyTorch, Ultralytics, OpenCV, and the trained model. Free hosting instances may sleep after periods of inactivity and may not be suitable for continuous real-time inference.

## Limitations

- The model detects visible waste in images; it does not directly measure water quality.
- Bounding-box coordinates are image pixels, not geographical coordinates.
- GPS integration, sensor processing, and heat-map generation require additional components.
- Dataset evaluation metrics do not guarantee identical performance in real-world environments.
- Inference speed depends on the hosting hardware, image dimensions, and server load.
- The API should be secured and configured appropriately before public production use.

## Future Enhancements

- Integration with GPS and boat navigation systems
- Integration with pH, turbidity, and TDS sensors
- Geotagged waste detection records
- Pollution heat-map visualization
- Live camera-stream processing
- Dashboard integration for monitoring pollution observations
- Production authentication and API access controls

## Author

**Jishnu M. S.**

GitHub: [Jishnu-ms](https://github.com/Jishnu-ms)

---

*Developed as an AI-based visual detection component for water pollution monitoring applications.*
