# IoT Anomaly Detection with Federated Learning

Anomaly detection on IoT network traffic using Federated Learning. Multiple simulated IoT devices train local autoencoder models on their private data; a central server aggregates them via FedAvg to build a global anomaly detector — without sharing raw traffic data.

**Dataset**: [EdgeIIoTSet](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot) — 14 attack types vs. normal IoT sensor readings across 63 features.

---

## Project Structure

```
Anomaly-Detection-IoT/
├── data/
│   ├── attack_traffic/        # 14 attack CSVs (downloaded)
│   ├── normal_traffic/        # IoT sensor CSVs by device (downloaded)
│   └── splits/                # Per-device train/val/test splits (generated)
│       ├── Distance/
│       ├── Flame_Sensor/
│       └── ...
├── src/
│   ├── simulation/            # Federated Learning simulation
│   │   ├── main.py            # Entry point
│   │   ├── server.py          # CentralServer (FedAvg aggregation)
│   │   ├── device.py          # SimulatedDevice (threaded local training)
│   │   ├── preprocessor.py    # IoTPreprocessor (encoding + scaling)
│   │   ├── windowing.py       # Sliding-window builder for sequence models
│   │   └── models/            # Autoencoder architectures
│   │       ├── vanilla.py     # Dense autoencoder
│   │       ├── lstm.py        # LSTM autoencoder
│   │       ├── conv1d.py      # Conv1D autoencoder
│   │       └── transformer.py # Transformer autoencoder
│   ├── models/                # Saved global model weights (output)
│   └── utils/
│       ├── download_data.py          # Step 1: download dataset from Kaggle
│       ├── prepare_device_splits.py  # Step 2: build per-device splits
│       ├── prepare_general_test.py   # Step 3: build general test set
│       └── data_analysis.ipynb       # EDA notebook
├── results/                   # Evaluation reports (output)
└── requirements.txt
```

---

## Setup

### 1. Create and activate the virtual environment

```bash
cd Anomaly-Detection-IoT
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Kaggle credentials

The download script uses the Kaggle API. Place your `kaggle.json` API token at `~/.config/kaggle/kaggle.json` (Linux/Mac) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows).

You can download the token from your [Kaggle account settings](https://www.kaggle.com/settings) under **API → Create New Token**.

---

## Step-by-Step Execution

### Step 1 — Download the dataset

Run from the repository root:

```bash
python src/utils/download_data.py
```

This downloads the EdgeIIoTSet dataset via `kagglehub`, removes large `.pcap` files to save disk space, and copies the CSVs into:

- `data/attack_traffic/` — 14 attack category CSVs
- `data/normal_traffic/` — IoT sensor reading CSVs (one subdirectory per device)

### Step 2 — Build per-device splits

Run from the repository root:

```bash
python src/utils/prepare_device_splits.py
```

For each of the 8 participating devices, this creates `data/splits/<device>/train.csv`, `val.csv`, and `test.csv`. The split strategy:

- **Train**: first 70% of normal rows (capped at 20% of the full file)
- **Val**: next 15% of normal rows
- **Test**: interleaved normal segments + assigned attack chunks (round-robin across devices)

Two devices are excluded from federated training: `Heart_Rate` and `Modbus`.

### Step 3 — Build the general test set

Run from the repository root:

```bash
python src/utils/prepare_general_test.py
```

Creates `data/splits/general_test.csv` — a single file covering all 14 attack types pooled from every device's normal traffic. Used to evaluate the final global model's generalisation.

### Step 4 — Run the EDA notebook (optional)

```bash
cd src/utils
jupyter notebook data_analysis.ipynb
```

The notebook must be run from `src/utils/` so its relative data paths resolve correctly. It explores the dataset schema, class balance, protocol distributions, and null patterns.

### Step 5 — Run the Federated Learning simulation

```bash
cd src/simulation
python main.py
```

The simulation must be run from `src/simulation/` because the module imports use bare names (`from device import ...`).

---

## Configuring the Simulation

Edit the constants at the top of `src/simulation/main.py` before running:

```python
MODEL_TYPE  = "vanilla"   # "vanilla" | "lstm" | "conv1d" | "transformer"
WINDOW_SIZE = None        # None for vanilla; e.g. 30 for lstm / conv1d / transformer

NUM_ROUNDS               = 7    # federated communication rounds
LOCAL_EPOCHS             = 20   # max local training epochs per round per device
EARLY_STOPPING_PATIENCE  = 3    # 0 to disable early stopping

DEVICE_NAMES = [          # devices to include; must match subdirs in data/splits/
    "Distance",
    "Flame_Sensor",
    "IR_Receiver",
    "phValue",
    "Soil_Moisture",
    "Sound_Sensor",
    "Temperature_and_Humidity",
    "Water_Level",
]
```

| `MODEL_TYPE`  | `WINDOW_SIZE` | Description                       |
|---------------|---------------|-----------------------------------|
| `"vanilla"`   | `None`        | Dense autoencoder, flat input     |
| `"lstm"`      | e.g. `30`     | LSTM autoencoder, sequence input  |
| `"conv1d"`    | e.g. `30`     | Conv1D autoencoder, sequence input|
| `"transformer"` | e.g. `30`   | Transformer autoencoder, sequence input |

---

## Outputs

After the simulation completes:

| Path | Contents |
|------|----------|
| `src/models/<model_type>_global_model.keras` | Final global model weights |
| `results/<model_type>_network_report.txt` | Aggregated per-device metrics (AUROC, F1, confusion matrix) |
| `results/<model_type>_global_report.txt` | General test set evaluation with per-attack-type breakdown |

### Example output (console)

```
Building global category vocabulary...
Computing input dimension from first device split...
  input_dim = 72

--- Starting Federated Learning Simulation ---

[Distance] Round 1/7 — local training ...
...
=== Network-Level Device Metrics ===
Devices: 8 | Samples: 12345 | AUROC(w): 0.9123 | ...

[Global Eval] Threshold: 0.0042 | AUROC: 0.8971 | PR-AUC: 0.7834 | ...

Per-Attack-Type Metrics:
  Attack Type                              AUROC  PR-AUC  Precision  Recall  ...
  -----------------------------------------------------------------------
  Backdoor_attack                         0.9500  0.8100     0.8800  0.9100  ...
  DDoS_HTTP_Flood_attack                  0.9800  0.9200     0.9300  0.9700  ...
  ...
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│              CentralServer                  │
│  - Holds global model weights (FedAvg)      │
│  - threading.Barrier fires aggregation      │
│    when all devices submit their update     │
└───────────────┬─────────────────────────────┘
                │  download weights / receive update
    ┌───────────┼───────────┐
    ▼           ▼           ▼
[Device 1]  [Device 2]  [Device N]   (SimulatedDevice threads)
 private     private     private
 data        data        data
```

Each `SimulatedDevice` runs as a `threading.Thread`. Per round it:
1. Downloads the current global weights from the server
2. Fits an autoencoder on its local training split
3. Sends updated weights back; the server aggregates once all devices report

Anomaly detection is reconstruction-based: the threshold is the 99th percentile of validation reconstruction errors (median MSE per sample). Test samples with error above the threshold are flagged as attacks.
