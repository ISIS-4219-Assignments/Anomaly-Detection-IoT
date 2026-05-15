"""generate_threshold.py

One-time script that computes the anomaly detection threshold from the
validation splits (pure normal traffic) and saves it alongside the
preprocessor cache.

The threshold is the 99th percentile of per-window reconstruction errors
when the global Conv1D model runs on normal-only validation data — exactly
replicating what each FL client computed locally during training (device.py:268).

Run once from the project root after generate_preprocessor.py:

    python deployment/generate_threshold.py

The result is stored in deployment/threshold_cache.pkl and loaded by the
Streamlit dashboard so inference uses a fixed, model-derived threshold
instead of recomputing one from whatever data is uploaded.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SIM_DIR = _PROJECT_ROOT / "src" / "simulation"
_MODEL_PATH = _PROJECT_ROOT / "src" / "models" / "conv1d_autoencoder_final.keras"
_PREPROCESSOR_PATH = Path(__file__).resolve().parent / "preprocessor_cache.pkl"
_OUTPUT_PATH = Path(__file__).resolve().parent / "threshold_cache.pkl"
_SPLITS_DIR = _PROJECT_ROOT / "data" / "splits"
_WINDOW_SIZE = 30

_DEVICE_NAMES = [
    "Distance",
    "Flame_Sensor",
    "IR_Receiver",
    "phValue",
    "Soil_Moisture",
    "Sound_Sensor",
    "Temperature_and_Humidity",
    "Water_Level",
]

sys.path.insert(0, str(_SIM_DIR))


def main() -> None:
    """Compute and persist the anomaly threshold from normal validation data."""
    if not _PREPROCESSOR_PATH.exists():
        print("ERROR: preprocessor_cache.pkl not found.")
        print("Run deployment/generate_preprocessor.py first.")
        sys.exit(1)

    val_paths = [_SPLITS_DIR / name / "val.csv" for name in _DEVICE_NAMES]
    missing = [p for p in val_paths if not p.exists()]
    if missing:
        print("Missing val splits:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    print("Loading preprocessor…")
    with open(_PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)["preprocessor"]

    print("Loading Conv1D model…")
    import keras
    model = keras.models.load_model(str(_MODEL_PATH))

    from windowing import create_windows

    all_errors: list[float] = []

    for device_name, val_path in zip(_DEVICE_NAMES, val_paths):
        print(f"  Processing {device_name}…", end=" ", flush=True)
        df = pd.read_csv(val_path, low_memory=False)

        try:
            X, _ = preprocessor.transform(df)
            X_arr = X.values.astype("float32")
        except Exception as exc:
            print(f"SKIP (preprocess error: {exc})")
            continue

        X_windows = create_windows(X_arr, _WINDOW_SIZE)
        if X_windows.shape[0] == 0:
            print("SKIP (no windows)")
            continue

        X_pred = model.predict(X_windows, batch_size=256, verbose=0)
        errors = np.median((X_windows - X_pred) ** 2, axis=(1, 2)).astype(float)
        all_errors.extend(errors.tolist())
        print(f"{len(errors):,} windows, max_err={errors.max():.6f}")

    if not all_errors:
        print("ERROR: no errors collected — threshold not saved.")
        sys.exit(1)

    all_errors_arr = np.array(all_errors, dtype=float)
    threshold = float(np.percentile(all_errors_arr, 99))

    print(f"\nThreshold (99th pct of {len(all_errors_arr):,} normal windows): {threshold:.8f}")

    with open(_OUTPUT_PATH, "wb") as f:
        pickle.dump({"threshold": threshold, "n_windows": len(all_errors_arr)}, f)

    print(f"Saved → {_OUTPUT_PATH}")
    print("Commit this file to the repository before deploying to Streamlit Cloud.")


if __name__ == "__main__":
    main()
