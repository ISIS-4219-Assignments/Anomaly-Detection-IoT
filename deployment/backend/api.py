"""api.py

FastAPI application exposing FL training control, WebSocket event streaming,
and inference endpoints.

Endpoints
---------
GET  /status          Current training state and round counter.
POST /train           Start a new FL run (returns 409 if one is in progress).
WS   /ws/training     Stream training events to the connected client.
POST /predict         Anomaly detection on an uploaded CSV file.
"""

import asyncio
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SIM_DIR = _PROJECT_ROOT / "src" / "simulation"
sys.path.insert(0, str(_SIM_DIR))

from training_runner import (
    PREPROCESSOR_CACHE_PATH,
    MODEL_WINDOW_SIZES,
    TrainingConfig,
    TrainingRunner,
)

_MODELS_DIR = _PROJECT_ROOT / "src" / "models"

runner = TrainingRunner()
app = FastAPI(title="FL Anomaly Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainingRequest(BaseModel):
    """Request body schema for POST /train."""

    model_type: str = "conv1d"
    num_rounds: int = 7
    local_epochs: int = 20
    early_stopping_patience: int = 3


@app.get("/status")
async def get_status() -> dict:
    """Return the current training status and round progress."""
    return {
        "status": runner.status,
        "current_round": runner.current_round,
        "model_type": runner.config.model_type if runner.config else None,
        "num_rounds": runner.config.num_rounds if runner.config else None,
    }


@app.post("/train")
async def start_training(request: TrainingRequest) -> dict:
    """Start a federated learning training run.

    Returns 409 Conflict if a training run is already in progress.
    The run executes in a daemon thread; progress is streamed via /ws/training.
    """
    if runner.is_running:
        raise HTTPException(status_code=409, detail="Training already in progress.")

    config = TrainingConfig(
        model_type=request.model_type,
        num_rounds=request.num_rounds,
        local_epochs=request.local_epochs,
        early_stopping_patience=request.early_stopping_patience,
    )
    runner.start(config, asyncio.get_running_loop())
    return {"status": "started", "model_type": config.model_type}


@app.websocket("/ws/training")
async def training_websocket(websocket: WebSocket) -> None:
    """Stream FL training events to the connected client.

    The connection stays open until training_complete is received or the client
    disconnects.  Each message is a JSON object with at minimum a 'type' field.
    """
    await websocket.accept()
    subscription: asyncio.Queue = runner.subscribe()

    try:
        while True:
            event = await asyncio.wait_for(subscription.get(), timeout=60.0)
            await websocket.send_json(event)
            if event.get("type") == "training_complete":
                break
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        runner.unsubscribe(subscription)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """Run anomaly detection on an uploaded CSV file.

    The global preprocessor must exist (created automatically after /train
    completes).  Returns per-sample MSE, binary anomaly flags, and the
    threshold used for classification.
    """
    if not PREPROCESSOR_CACHE_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                "Preprocessor not found. Run /train at least once so the "
                "global preprocessor is fitted and cached."
            ),
        )

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")), low_memory=False)

    with open(PREPROCESSOR_CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    preprocessor = cache["preprocessor"]

    X, _ = preprocessor.transform(df)
    X_arr = X.values.astype("float32")

    model_type = runner.config.model_type if runner.config else "conv1d"
    window_size = MODEL_WINDOW_SIZES.get(model_type)

    if window_size is not None:
        from windowing import create_windows
        X_arr = create_windows(X_arr, window_size)

    model_path = _MODELS_DIR / f"{model_type}_autoencoder_final.keras"
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Trained model not found: {model_path.name}",
        )

    import keras
    model = keras.models.load_model(str(model_path))
    X_pred = model.predict(X_arr, batch_size=256, verbose=0)

    axes = tuple(range(1, X_arr.ndim))
    errors = np.median((X_arr - X_pred) ** 2, axis=axes).astype(float)
    threshold = float(np.percentile(errors, 99))
    anomaly_flags = (errors > threshold).astype(int).tolist()

    return {
        "mse": errors.tolist(),
        "anomaly": anomaly_flags,
        "threshold": threshold,
        "n_samples": int(len(errors)),
        "n_anomalies": int(sum(anomaly_flags)),
    }
