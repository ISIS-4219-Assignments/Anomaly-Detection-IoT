"""generate_preprocessor.py

One-time script that fits the global IoTPreprocessor on all device training
splits and saves the result to deployment/preprocessor_cache.pkl.

Run once from the project root after downloading the dataset:

    python deployment/generate_preprocessor.py

The generated file must be committed to the repository so that the Streamlit
app can load it without requiring the raw training data.
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SIM_DIR = _PROJECT_ROOT / "src" / "simulation"
_SPLITS_DIR = _PROJECT_ROOT / "data" / "splits"
_OUTPUT_PATH = Path(__file__).resolve().parent / "preprocessor_cache.pkl"

sys.path.insert(0, str(_SIM_DIR))

from preprocessor import IoTPreprocessor, build_global_categories

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


def main() -> None:
    """Fit and persist the global preprocessor from all device training splits."""
    train_paths = [str(_SPLITS_DIR / name / "train.csv") for name in _DEVICE_NAMES]

    missing = [p for p in train_paths if not Path(p).exists()]
    if missing:
        print("Missing splits:")
        for p in missing:
            print(f"  {p}")
        print("\nRun src/utils/prepare_device_splits.py first.")
        sys.exit(1)

    print("Building global category vocabulary...")
    known_categories = build_global_categories(train_paths)

    print("Fitting global preprocessor on pooled training data...")
    pooled = pd.concat(
        [pd.read_csv(p, low_memory=False) for p in train_paths],
        ignore_index=True,
    )
    preprocessor = IoTPreprocessor(known_categories=known_categories)
    preprocessor.fit_transform(pooled)

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_PATH, "wb") as f:
        pickle.dump({"preprocessor": preprocessor, "known_categories": known_categories}, f)

    print(f"\nSaved → {_OUTPUT_PATH}")
    print("Commit this file to the repository before deploying to Streamlit Cloud.")


if __name__ == "__main__":
    main()
