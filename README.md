# IoT Anomaly Detection with Federated Learning

Unsupervised anomaly detection on IoT network traffic using Federated Learning. Eight simulated IoT devices each train a local autoencoder on their private normal traffic; a central server aggregates the weights via sample-weighted FedAvg to produce a global anomaly detector — without sharing raw data.

**Dataset**: [EdgeIIoTSet](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot) — 14 attack types vs. normal IoT sensor readings across 63 features.

---

## Project Structure

```
Anomaly-Detection-IoT/
├── ANALYSIS.md                          # Experiment results and model comparison
├── README.md
├── requirements.txt
├── data/
│   ├── attack_traffic/                  # 14 attack CSVs             (git-ignored, downloaded)
│   ├── normal_traffic/                  # IoT sensor CSVs by device  (git-ignored, downloaded)
│   └── splits/                          # Train/val/test splits       (git-ignored, generated)
│       ├── general_test.csv
│       ├── Distance/
│       │   ├── train.csv
│       │   ├── val.csv
│       │   └── test.csv
│       └── ...
├── src/
│   ├── models/                          # Saved global model weights  (output)
│   │   └── <model_type>_autoencoder_final.keras
│   ├── simulation/                      # Federated Learning simulation
│   │   ├── main.py                      # Entry point and evaluation
│   │   ├── server.py                    # CentralServer — FedAvg aggregation
│   │   ├── device.py                    # SimulatedDevice — threaded local training
│   │   ├── preprocessor.py              # IoTPreprocessor — encoding + scaling
│   │   ├── windowing.py                 # Sliding-window builder for sequence models
│   │   └── models/                      # Autoencoder architectures
│   │       ├── factory.py               # build_model() dispatch
│   │       ├── vanilla.py               # Dense autoencoder
│   │       ├── lstm.py                  # LSTM autoencoder
│   │       ├── conv1d.py                # Conv1D autoencoder
│   │       └── transformer.py           # Transformer autoencoder
│   └── utils/
│       ├── download_data.py             # Step 1: download dataset from Kaggle
│       ├── prepare_device_splits.py     # Step 2: build per-device splits
│       ├── prepare_general_test.py      # Step 3: build general test set
│       └── data_analysis.ipynb          # EDA notebook
└── results/                             # Evaluation reports and loss curves (output)
    ├── <model_type>_network_report.txt
    ├── <model_type>_global_report.txt
    └── <device>/
        ├── <model_type>_report.txt
        └── <model_type>_loss_curve.png
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

---

## Step-by-Step Execution

### Step 1 — Download the dataset

The download script uses the Kaggle API. Before running it, place your `kaggle.json` token at `~/.config/kaggle/kaggle.json` (Linux/Mac) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows). You can generate one from your [Kaggle account settings](https://www.kaggle.com/settings) under **API → Create New Token**.

Then run from the repository root:

```bash
python src/utils/download_data.py
```

This downloads the EdgeIIoTSet dataset via `kagglehub`, removes large `.pcap` files, and copies the CSVs into:

- `data/attack_traffic/` — 14 attack category CSVs
- `data/normal_traffic/` — IoT sensor CSVs (one subdirectory per device)

### Step 2 — Build per-device splits

```bash
python src/utils/prepare_device_splits.py
```

For each of the 8 participating devices, creates `data/splits/<device>/train.csv`, `val.csv`, and `test.csv`. Split strategy:

- **Train**: first 70% of normal rows (capped at 20% of the full file)
- **Val**: next 15% of normal rows
- **Test**: interleaved normal segments + assigned attack chunks (round-robin across devices)

`Heart_Rate` and `Modbus` are excluded — they contain no usable attack assignments.

### Step 3 — Build the general test set

```bash
python src/utils/prepare_general_test.py
```

Produces `data/splits/general_test.csv` — a single file covering all 14 attack types interleaved with normal traffic pooled from all 8 devices. Used to evaluate the final global model's generalisation across the full threat landscape.

### Step 4 — Run the EDA notebook (optional)

```bash
cd src/utils
jupyter notebook data_analysis.ipynb
```

Must be run from `src/utils/` so its relative data paths resolve correctly. Explores the dataset schema, class balance, protocol distributions, and null patterns.

### Step 5 — Run the Federated Learning simulation

```bash
cd src/simulation
python main.py
```

Must be run from `src/simulation/` because the module imports use bare names (`from device import ...`).

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

| `MODEL_TYPE`    | `WINDOW_SIZE` | Architecture                               |
|-----------------|---------------|--------------------------------------------|
| `"vanilla"`     | `None`        | Dense autoencoder, flat input              |
| `"lstm"`        | e.g. `30`     | LSTM autoencoder, sequence input           |
| `"conv1d"`      | e.g. `30`     | Conv1D autoencoder, sequence input         |
| `"transformer"` | e.g. `30`     | Transformer autoencoder, sequence input    |

---

## Outputs

After the simulation completes:

| Path | Contents |
|------|----------|
| `src/models/<model_type>_autoencoder_final.keras` | Final global model weights |
| `results/<device>/<model_type>_report.txt` | Per-device threshold, AUROC, F1, confusion matrix |
| `results/<device>/<model_type>_loss_curve.png` | Per-device train/val loss across all rounds |
| `results/<model_type>_network_report.txt` | Aggregated metrics across all 8 devices |
| `results/<model_type>_global_report.txt` | General test set evaluation with per-attack-type breakdown |

### Example output (console)

```
Building global category vocabulary...
Computing input dimension from first device split...
  input_dim = 72

--- Starting Federated Learning Simulation ---

[Device Distance] Loading and preprocessing data...
[Device Distance] Ready. train=(1196, 30, 72), val=(256, 30, 72)
...
[Device Distance] Round 1 — building model and loading global weights...
[Device Distance] Round 1 — loss: 0.0023, val_loss: 0.0019. Sending weights to server...
[Server] Received update from Device Distance (n=1196). (1/8)
...
--- Aggregating models for Round 1 ---
[Server] Global model updated after round 1.

...

All threads closed.

=== Network-Level Device Metrics ===
Devices: 8 | Samples: 348865 | AUROC(w): 1.0000 | PR-AUC(w): 0.9992 | Recall: 1.0000 | F1: 0.9638

[Global Eval] Fitting global preprocessor on pooled training data...
[Global Eval] Threshold: 0.001923
[Global Eval] Threshold: 0.0019 | AUROC: 0.9999 | PR-AUC: 0.9996 | Precision: 0.9831 | Recall: 0.9995 | ...

Per-Attack-Type Metrics:
  Attack Type                              AUROC  PR-AUC  Precision   Recall  ...
  -----------------------------------------------------------------------------------
  Backdoor_attack                         1.0000  1.0000     0.9474   1.0000  ...
  DDoS_HTTP_Flood_attack                  0.9968  0.9945     0.9766   0.9868  ...
  ...

[Server] Global model saved → .../src/models/conv1d_autoencoder_final.keras
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
                │  broadcast weights / collect updates
    ┌───────────┼───────────┐
    ▼           ▼           ▼
[Device 1]  [Device 2]  [Device N]   (SimulatedDevice threads)
 private     private     private
 data        data        data
```

Each `SimulatedDevice` runs as a `threading.Thread`. Per round it:

1. Downloads the current global weights from the server
2. Trains an autoencoder locally on its private normal-traffic split
3. Sends updated weights back; the server aggregates via weighted FedAvg once all devices report

After all rounds, each device computes its anomaly threshold (99th percentile of per-sample validation reconstruction errors), evaluates the global model on its test split, and saves its report and loss curve.

Anomaly detection is reconstruction-based: a sample's anomaly score is the median squared error between input and reconstruction. Samples with error above the threshold are flagged as attacks.

### Model architectures

All four models are autoencoders trained on normal traffic only (target = input, loss = MSE).

| Architecture | Input shape | Encoder bottleneck | Key design choice |
|---|---|---|---|
| Vanilla | `(batch, features)` | Dense 64→32→16 | No temporal context |
| LSTM | `(batch, window, features)` | LSTM(64)→LSTM(32)→LSTM(32) | Recurrent state captures ordering |
| Conv1D | `(batch, window, features)` | Conv(64)→Conv(32)→GAP→Dense(16) | GlobalAveragePooling avoids XLA issues |
| Transformer | `(batch, window, features)` | Embed→PosEnc→MHA→FFN→GAP→Dense(16) | Learned positional encoding |
