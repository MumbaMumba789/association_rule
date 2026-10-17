
from pathlib import Path
import json

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# --------------------------------------------------
# 2. Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="DBSCAN Model Serving API",
    description="API for serving versioned DBSCAN clustering models.",
    version="1.0.0",
)


# --------------------------------------------------
# 3. Load model versions and metadata
# --------------------------------------------------

MODELS = {}

for version in ("v1", "v2"):
    model_path = MODEL_DIR / f"dbscan_{version}.joblib"
    metadata_path = MODEL_DIR / f"dbscan_{version}_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {metadata_path}"
        )

    MODELS[version] = {
        "model": joblib.load(model_path),
        "metadata": json.loads(
            metadata_path.read_text(encoding="utf-8")
        ),
    }


# --------------------------------------------------
# 4. Load scaler and saved training data
# --------------------------------------------------

SCALER_PATH = MODEL_DIR / "dbscan_scaler_v1.joblib"
TRAINING_DATA_PATH = MODEL_DIR / "dbscan_training_data.npz"

if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Missing scaler: {SCALER_PATH}")

if not TRAINING_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Missing training data: {TRAINING_DATA_PATH}. "
        "Run the notebook cell that saves X_scaled, labels, "
        "and labels_v2."
    )

scaler = joblib.load(SCALER_PATH)
training_data = np.load(TRAINING_DATA_PATH)

X_scaled = training_data["X_scaled"]

LABELS_BY_VERSION = {
    "v1": training_data["labels_v1"],
    "v2": training_data["labels_v2"],
}


# --------------------------------------------------
# 5. Request schema
# --------------------------------------------------

class PointRequest(BaseModel):
    x: float = Field(description="Original x coordinate")
    y: float = Field(description="Original y coordinate")
    version: str = Field(
        default="v1",
        description="Model version: v1 or v2",
    )


# --------------------------------------------------
# 6. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "DBSCAN Model Serving API",
        "available_versions": list(MODELS.keys()),
    }


# --------------------------------------------------
# 7. Model metadata endpoint
# --------------------------------------------------

@app.get("/models/{version}")
def model_info(version: str):
    if version not in MODELS:
        raise HTTPException(
            status_code=404,
            detail="Model version not found",
        )

    return MODELS[version]["metadata"]


# --------------------------------------------------
# 8. Clustering endpoint
# --------------------------------------------------

@app.post("/cluster")
def get_cluster(request: PointRequest):
    if request.version not in MODELS:
        raise HTTPException(
            status_code=404,
            detail="Model version not found",
        )

    # Transform the new point using the saved scaler.
    point = np.array([[request.x, request.y]])
    point_scaled = scaler.transform(point)

    # Find the nearest training observation.
    distances = np.linalg.norm(
        X_scaled - point_scaled,
        axis=1,
    )

    nearest_index = int(np.argmin(distances))
    nearest_distance = float(distances[nearest_index])

    model = MODELS[request.version]["model"]
    labels = LABELS_BY_VERSION[request.version]

    # Only return a label when the nearest point is
    # within this model version's eps threshold.
    if nearest_distance > model.eps:
        return {
            "version": request.version,
            "cluster": None,
            "nearest_distance": round(nearest_distance, 4),
            "method": "nearest training point",
            "message": "No nearby training point within eps.",
        }

    cluster = int(labels[nearest_index])

    return {
        "version": request.version,
        "cluster": cluster,
        "nearest_distance": round(nearest_distance, 4),
        "method": "nearest training point",
        "is_noise": cluster == -1,
    }
