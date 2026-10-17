from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.metrics import pairwise_distances

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

app = FastAPI(title="DBSCAN Model Serving API")

# Available model versions
MODELS = {
    "v1": {
        "model": joblib.load(MODEL_DIR / "dbscan_v1.joblib"),
        "metadata": json.loads(
            (MODEL_DIR / "dbscan_v1_metadata.json").read_text()
        ),
    },
    "v2": {
        "model": joblib.load(MODEL_DIR / "dbscan_v2.joblib"),
        "metadata": json.loads(
            (MODEL_DIR / "dbscan_v2_metadata.json").read_text()
        ),
    },
}

training_data = np.load(MODEL_DIR / "dbscan_training_data.npz")

X_scaled = training_data["X_scaled"]
labels_by_version = {
    "v1": training_data["labels_v1"],
    "v2": training_data["labels_v2"],
}

class PointRequest(BaseModel):
    x: float = Field(description="Original x coordinate")
    y: float = Field(description="Original y coordinate")
    version: str = "v1"


@app.get("/")
def home():
    return {
        "message": "DBSCAN Model Serving API",
        "available_versions": list(MODELS.keys()),
    }


@app.get("/models/{version}")
def model_info(version: str):
    if version not in MODELS:
        raise HTTPException(
            status_code=404,
            detail="Model version not found",
        )

    return MODELS[version]["metadata"]


@app.post("/cluster")
def get_cluster(request: PointRequest):
    if request.version not in MODELS:
        raise HTTPException(
            status_code=404,
            detail="Model version not found",
        )

    model = MODELS[request.version]["model"]
    scaler = joblib.load(MODEL_DIR / "dbscan_scaler_v1.joblib")

    point = np.array([[request.x, request.y]])
    point_scaled = scaler.transform(point)

    distances = pairwise_distances(point_scaled, X_scaled)[0]
    nearest_index = int(np.argmin(distances))
    nearest_distance = float(distances[nearest_index])

    if nearest_distance > model.eps:
        return {
            "version": request.version,
            "cluster": None,
            "message": "No nearby training point within eps.",
            "nearest_distance": round(nearest_distance, 4),
        }

    cluster = int(labels_by_version[request.version][nearest_index])

    return {
        "version": request.version,
        "cluster": cluster,
        "nearest_distance": round(nearest_distance, 4),
        "method": "nearest training point",
    }