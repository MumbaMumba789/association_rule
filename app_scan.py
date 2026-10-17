
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import time

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
LOG_FILE = BASE_DIR / "monitoring_log.csv"


# --------------------------------------------------
# 2. CREATE THE API
# --------------------------------------------------

app = FastAPI(
    title="DBSCAN Model Serving API",
    description="DBSCAN clustering API with model monitoring",
    version="1.0.0",
)


# --------------------------------------------------
# 3. LOAD MODEL VERSIONS AND METADATA
# --------------------------------------------------

MODELS = {}
METADATA = {}

for version in ["v1", "v2"]:
    model_path = MODEL_DIR / f"dbscan_{version}.joblib"
    metadata_path = MODEL_DIR / f"dbscan_{version}_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    MODELS[version] = joblib.load(model_path)

    with open(metadata_path, "r", encoding="utf-8") as file:
        METADATA[version] = json.load(file)


# --------------------------------------------------
# 4. LOAD THE SCALER AND TRAINING DATA
# --------------------------------------------------

scaler_path = MODEL_DIR / "dbscan_scaler_v1.joblib"
training_data_path = MODEL_DIR / "dbscan_training_data.npz"

scaler = joblib.load(scaler_path)

training_data = np.load(training_data_path)

X_scaled = training_data["X_scaled"]

LABELS_BY_VERSION = {
    "v1": training_data["labels_v1"],
    "v2": training_data["labels_v2"],
}


# --------------------------------------------------
# 5. DEFINE THE REQUEST FORMAT
# --------------------------------------------------

class PointRequest(BaseModel):
    x: float = Field(description="First coordinate")
    y: float = Field(description="Second coordinate")
    version: str = Field(
        default="v1",
        description="Model version: v1 or v2",
    )


# --------------------------------------------------
# 6. MONITORING LOG FUNCTION
# --------------------------------------------------

def log_request(
    version,
    x,
    y,
    cluster,
    is_noise,
    latency_ms,
    status,
    error=None,
):
    file_exists = LOG_FILE.exists()

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "version",
                "x",
                "y",
                "cluster",
                "is_noise",
                "latency_ms",
                "status",
                "error",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": version,
            "x": x,
            "y": y,
            "cluster": cluster,
            "is_noise": is_noise,
            "latency_ms": round(latency_ms, 2),
            "status": status,
            "error": error,
        })


# --------------------------------------------------
# 7. HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "DBSCAN Model Serving API is running",
        "available_versions": list(MODELS.keys()),
        "monitoring": "Enabled",
    }


# --------------------------------------------------
# 8. MODEL INFORMATION ENDPOINT
# --------------------------------------------------

@app.get("/models/{version}")
def get_model_info(version: str):

    if version not in METADATA:
        raise HTTPException(
            status_code=404,
            detail="Model version not found",
        )

    return METADATA[version]


# --------------------------------------------------
# 9. CLUSTERING ENDPOINT WITH MONITORING
# --------------------------------------------------

@app.post("/cluster")
def cluster_point(request: PointRequest):

    start_time = time.perf_counter()

    version = request.version
    x = request.x
    y = request.y

    cluster_label = None
    is_noise = None

    try:
        # Check that the requested model exists.
        if version not in MODELS:
            raise HTTPException(
                status_code=400,
                detail="Invalid model version. Use v1 or v2.",
            )

        model = MODELS[version]

        # Scale the incoming coordinates using the saved scaler.
        point = np.array([[x, y]])
        point_scaled = scaler.transform(point)

        # Find the nearest point from the training dataset.
        distances = np.linalg.norm(
            X_scaled - point_scaled,
            axis=1,
        )

        nearest_index = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_index])

        # This is a simple approximation, not native DBSCAN prediction.
        if nearest_distance > model.eps:
            cluster_label = None
            is_noise = None

            message = (
                "Point is too far from the nearest training point "
                "for this approximation."
            )

        else:
            cluster_label = int(
                LABELS_BY_VERSION[version][nearest_index]
            )

            is_noise = cluster_label == -1

            message = "Nearest training point used for approximation."

        # Calculate request processing time.
        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        # Record the successful API request.
        log_request(
            version=version,
            x=x,
            y=y,
            cluster=cluster_label,
            is_noise=is_noise,
            latency_ms=latency_ms,
            status="success",
        )

        return {
            "version": version,
            "input": {"x": x, "y": y},
            "cluster": cluster_label,
            "is_noise": is_noise,
            "nearest_distance": round(nearest_distance, 4),
            "latency_ms": round(latency_ms, 2),
            "method": "nearest_training_point_approximation",
            "message": message,
        }

    except HTTPException as exc:
        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        log_request(
            version=version,
            x=x,
            y=y,
            cluster=None,
            is_noise=None,
            latency_ms=latency_ms,
            status="error",
            error=str(exc.detail),
        )

        raise

    except Exception as exc:
        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        log_request(
            version=version,
            x=x,
            y=y,
            cluster=None,
            is_noise=None,
            latency_ms=latency_ms,
            status="error",
            error=str(exc),
        )

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the point.",
        ) from exc