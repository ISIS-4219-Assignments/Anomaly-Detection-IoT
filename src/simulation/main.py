"""
main.py
-------

Entry point for the Federated Learning simulation.

Swapping between model architectures
-------------------------------------
Change the two constants below and rerun.  Nothing else needs to change.

    MODEL_TYPE  = "vanilla"     →  Dense autoencoder,        flat input
    MODEL_TYPE  = "lstm"        →  LSTM autoencoder,          sequence input
    MODEL_TYPE  = "conv1d"      →  Conv1D autoencoder,        sequence input
    MODEL_TYPE  = "transformer" →  Transformer autoencoder,   sequence input

    WINDOW_SIZE = None          →  required for "vanilla"
    WINDOW_SIZE = 30            →  required for "lstm", "conv1d", and "transformer"

Run from this directory
-----------------------
    cd src/simulation
    python main.py
"""


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress XLA/TF C++ info logs

from preprocessor import IoTPreprocessor, build_global_categories
from device import SimulatedDevice
from server import CentralServer
from pathlib import Path
import pandas as pd
import threading
import os


# ---------------------------------------------------------------------------
# Simulation configuration — edit here to change behaviour
# ---------------------------------------------------------------------------


MODEL_TYPE  = "lstm"  # "vanilla" | "lstm" | "conv1d" | "transformer"
WINDOW_SIZE = 30       # None for vanilla; e.g. 30 for lstm / conv1d

NUM_ROUNDS    = 2   # federated communication rounds
LOCAL_EPOCHS  = 5   # local training epochs per round per device

# Devices whose splits will participate in this run.
# Each name must match a subdirectory under data/splits/.
DEVICE_NAMES = [
    "Distance",
    "Flame_Sensor",
    #"IR_Receiver",
    #"phValue",
]

# Absolute dir of src/simulation/
BASE_DIR = Path(__file__).resolve().parent

# Project Root dir
PROJECT_ROOT = BASE_DIR.parent.parent

# data/splits
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# src/models
MODELS_DIR = PROJECT_ROOT / "src" / "models"

# results/
RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_paths(device_name: str) -> dict[str, str]:

    """Return the train / val / test CSV paths for a given device.

    Parameters
    ----------
    device_name : str
        Name of the device directory under ``SPLITS_DIR``.

    Returns
    -------
    dict[str, str]
        ``{"train": ..., "val": ..., "test": ...}`` with absolute-style
        paths relative to the working directory.
    """

    base = os.path.join(SPLITS_DIR, device_name)
    return {
        "train": os.path.join(base, "train.csv"),
        "val":   os.path.join(base, "val.csv"),
        "test":  os.path.join(base, "test.csv"),
    }


def _compute_input_dim(train_path: str, known_categories: dict) -> int:

    """Derive the feature count after preprocessing one training split.

    A temporary :class:`~preprocessor.IoTPreprocessor` is fitted on the
    given file.  Its output column count becomes the ``input_dim`` shared
    by the server and all devices.

    Parameters
    ----------
    train_path : str
        Path to any device's training CSV.
    known_categories : dict[str, list]
        Global category vocabulary from :func:`~preprocessor.build_global_categories`.

    Returns
    -------
    int
        Number of feature columns after cleaning, encoding, and scaling.
    """

    pre = IoTPreprocessor(known_categories = known_categories)
    X, _ = pre.fit_transform(pd.read_csv(train_path, low_memory = False))
    return X.shape[1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:

    """Set up and run the Federated Learning simulation.

    Steps
    -----
    1. Build the global category vocabulary from all training splits so every
       device's one-hot encoding covers the same set of categories.
    2. Preprocess one split to determine ``input_dim`` (the number of features
       after encoding and scaling).  This value is the same for all devices
       because the preprocessor guarantees a uniform feature space.
    3. Initialise the :class:`~server.CentralServer` with the model
       architecture and start-of-training global weights.
    4. Create one :class:`~device.SimulatedDevice` thread per device, each
       pointing at its own private data split.
    5. Start all threads and wait for them to finish.
    """
    
    all_paths = [_make_paths(name) for name in DEVICE_NAMES]
    all_train_paths = [p["train"] for p in all_paths]

    # --- Step 1: global vocabulary ---
    print("Building global category vocabulary...")
    known_categories = build_global_categories(all_train_paths)

    # --- Step 2: feature dimension ---
    print("Computing input dimension from first device split...")
    input_dim = _compute_input_dim(all_train_paths[0], known_categories)
    print(f"  input_dim = {input_dim}\n")

    # --- Step 3: server ---
    server = CentralServer(
        total_clients     = len(DEVICE_NAMES),
        rounds_to_simulate= NUM_ROUNDS,
        model_type        = MODEL_TYPE,
        input_dim         = input_dim,
        window_size       = WINDOW_SIZE,
    )

    # --- Step 4: devices ---
    gpu_lock = threading.Lock()
    devices = [
        SimulatedDevice(
            client_id        = name,
            server           = server,
            data_paths       = paths,
            model_type       = MODEL_TYPE,
            input_dim        = input_dim,
            window_size      = WINDOW_SIZE,
            known_categories = known_categories,
            local_epochs     = LOCAL_EPOCHS,
            gpu_lock         = gpu_lock,
            results_dir      = RESULTS_DIR,
        )
        for name, paths in zip(DEVICE_NAMES, all_paths)
    ]

    # --- Step 5: run ---
    print("--- Starting Federated Learning Simulation ---\n")

    for device in devices:
        device.start()

    for device in devices:
        device.join()

    print("\nAll threads closed.")

    # Save the final global model to src/models/
    server.save_global_model(MODELS_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
