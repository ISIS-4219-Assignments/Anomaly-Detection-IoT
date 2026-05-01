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

from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)
from preprocessor import IoTPreprocessor, build_global_categories
from models.factory import build_model
from windowing import create_windows
from device import SimulatedDevice
from server import CentralServer
from pathlib import Path
import numpy as np
import pandas as pd
import threading
import os


# ---------------------------------------------------------------------------
# Simulation configuration — edit here to change behaviour
# ---------------------------------------------------------------------------


MODEL_TYPE  = "vanilla"  # "vanilla" | "lstm" | "conv1d" | "transformer"
WINDOW_SIZE = None       # None for vanilla; e.g. 30 for lstm / conv1d / transformer

NUM_ROUNDS    = 3   # federated communication rounds
LOCAL_EPOCHS  = 5   # local training epochs per round per device (max)
EARLY_STOPPING_PATIENCE = 3  # stop early if val_loss stalls; 0 to disable

# Devices whose splits will participate in this run.
# Each name must match a subdirectory under data/splits/.
DEVICE_NAMES = [
    "Distance",
    "Flame_Sensor",
    "IR_Receiver",
    "phValue",
    "Soil_Moisture",
    "Sound_Sensor",
    "Temperature_and_Humidity",
    "Water_Level"
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

# data/splits/general_test.csv  (built by src/utils/prepare_general_test.py)
GENERAL_TEST_PATH = SPLITS_DIR / "general_test.csv"


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
        ``{"train": ..., "val": ..., "test": ...}`` with absolute paths.
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
# Global evaluation
# ---------------------------------------------------------------------------


def _evaluate_global_test(
    global_weights: list,
    all_train_paths: list[str],
    all_val_paths: list[str],
    known_categories: dict,
    model_type: str,
    input_dim: int,
    window_size: int | None,
    general_test_path: str,
    results_dir,
    gpu_lock: threading.Lock | None = None,
) -> None:

    """Evaluate the final global model on the general test set.

    Fits a new :class:`~preprocessor.IoTPreprocessor` on pooled training data
    from all devices (giving a global scaler), derives the anomaly threshold
    from the 99th percentile of pooled validation reconstruction errors, then
    runs inference on ``general_test_path``.

    Reports overall AUROC / Precision / Recall / F1 and a per-attack-type
    recall breakdown.  Saves a text report to
    ``<results_dir>/<model_type>_global_report.txt``.

    Skips gracefully if ``general_test_path`` does not exist, printing a
    reminder to run ``prepare_general_test.py`` first.

    Parameters
    ----------
    global_weights : list[np.ndarray]
        Final global model weights from the server.
    all_train_paths : list[str]
        Training CSV paths for every device (used to fit the global scaler).
    all_val_paths : list[str]
        Validation CSV paths for every device (used to compute the threshold).
    known_categories : dict[str, list]
        Global categorical vocabulary built before training.
    model_type : str
        Architecture name — one of ``"vanilla"``, ``"lstm"``, ``"conv1d"``,
        ``"transformer"``.
    input_dim : int
        Feature dimension after preprocessing.
    window_size : int or None
        Sliding-window size; ``None`` for the vanilla model.
    general_test_path : str
        Path to the general test CSV produced by ``prepare_general_test.py``.
    results_dir : Path or str or None
        Directory where the report will be written.  Skips saving if ``None``.
    gpu_lock : threading.Lock or None
        Shared lock used to serialise Keras inference across threads.
    """

    if not os.path.isfile(general_test_path):
        print(
            f"\n[Global Eval] General test file not found: {general_test_path}\n"
            "[Global Eval] Run  python src/utils/prepare_general_test.py  first."
        )
        return

    # --- Fit a global preprocessor on pooled training data ---
    print("\n[Global Eval] Fitting global preprocessor on pooled training data...")
    pooled_train = pd.concat(
        [pd.read_csv(p, low_memory=False) for p in all_train_paths],
        ignore_index=True,
    )
    pre = IoTPreprocessor(known_categories=known_categories)
    pre.fit_transform(pooled_train)

    # --- Compute threshold from pooled validation errors ---
    print("[Global Eval] Computing threshold from pooled validation data...")
    val_arrays = [
        pre.transform(pd.read_csv(p, low_memory=False))[0].values.astype("float32")
        for p in all_val_paths
    ]
    X_val = np.concatenate(val_arrays, axis=0)
    if window_size is not None:
        X_val = create_windows(X_val, window_size)

    lock  = gpu_lock or threading.Lock()
    model = build_model(model_type, input_dim, window_size)
    model.set_weights(global_weights)

    with lock:
        X_val_pred = model.predict(X_val, batch_size=256, verbose=0)

    axes       = tuple(range(1, X_val.ndim))
    val_errors = np.median((X_val - X_val_pred) ** 2, axis=axes)
    threshold  = float(np.percentile(val_errors, 99))
    print(f"[Global Eval] Threshold: {threshold:.6f}")

    # --- Evaluate on general test set ---
    print("[Global Eval] Evaluating on general test set...")
    test_df              = pd.read_csv(general_test_path, low_memory=False)
    X_test, y_test_df   = pre.transform(test_df)
    X_test               = X_test.values.astype("float32")
    y_labels             = y_test_df["Attack_label"].values
    attack_types         = (y_test_df["Attack_type"].values
                            if "Attack_type" in y_test_df.columns else None)

    if window_size is not None:
        X_test       = create_windows(X_test, window_size)
        y_labels     = y_labels[window_size - 1:]
        if attack_types is not None:
            attack_types = attack_types[window_size - 1:]

    with lock:
        X_pred = model.predict(X_test, batch_size=256, verbose=0)

    axes   = tuple(range(1, X_test.ndim))
    errors = np.median((X_test - X_pred) ** 2, axis=axes)
    y_pred = (errors > threshold).astype(int)

    normal_mask = y_labels == 0
    attack_mask = y_labels == 1

    normal_mse  = float(np.median(errors[normal_mask])) if normal_mask.any() else float("nan")
    attack_mse  = float(np.median(errors[attack_mask])) if attack_mask.any() else float("nan")
    precision   = precision_score(y_labels, y_pred, zero_division=0)
    recall_val  = recall_score(y_labels, y_pred, zero_division=0)
    cm          = confusion_matrix(y_labels, y_pred)
    auroc       = (roc_auc_score(y_labels, errors)
                   if normal_mask.any() and attack_mask.any() else float("nan"))
    prauc       = (average_precision_score(y_labels, errors)
                   if normal_mask.any() and attack_mask.any() else float("nan"))
    bal_acc     = balanced_accuracy_score(y_labels, y_pred)

    tp_g = int(cm[1, 1])
    fn_g = int(cm[1, 0])
    n_g  = len(y_labels)
    rng_g        = np.random.default_rng(42)
    attack_idx_g = np.where(attack_mask)[0]
    normal_idx_g = np.where(normal_mask)[0]
    if attack_mask.any() and normal_mask.any():
        bal_size_g = min(len(attack_idx_g), len(normal_idx_g))
        bal_idx_g  = np.concatenate([
            attack_idx_g,
            rng_g.choice(normal_idx_g, size=bal_size_g, replace=False),
        ])
        f1_bal_g = f1_score(y_labels[bal_idx_g], y_pred[bal_idx_g], zero_division=0)
    else:
        f1_bal_g = float("nan")

    print(
        f"[Global Eval] Threshold: {threshold:.4f} | "
        f"AUROC: {auroc:.4f} | PR-AUC: {prauc:.4f} | "
        f"Precision: {precision:.4f} | Recall: {recall_val:.4f} | "
        f"Acc(bal): {bal_acc:.4f} | F1(bal): {f1_bal_g:.4f} | "
        f"Normal Median MSE: {normal_mse:.6f} | "
        f"Attack Median MSE: {attack_mse:.6f} | "
        f"TP: {tp_g} | FN: {fn_g} | n: {n_g}"
    )

    # --- Per-attack-type breakdown ---
    # Threshold-free metrics (AUROC, PR-AUC): computed on all normal rows + attack type.
    # Threshold-based metrics (Precision, Recall, Acc(bal), F1(bal)): computed on a 1:1
    # balanced subset (equal attack and sampled normal rows) to avoid class-imbalance bias.
    rng = np.random.default_rng(42)
    normal_idx = np.where(normal_mask)[0]

    per_type_lines = []
    if attack_types is not None:
        header = (
            f"  {'Attack Type':<40} {'AUROC':>7} {'PR-AUC':>7} "
            f"{'Precision':>10} {'Recall':>7} {'Acc(bal)':>9} {'F1(bal)':>8} {'Median MSE':>14} {'TP':>7} {'FN':>7} {'n':>7}"
        )
        separator = "  " + "-" * (len(header) - 2)
        per_type_lines += [header, separator]

        for atype in sorted(set(attack_types[attack_mask]) if attack_mask.any() else []):
            atk_mask = attack_types == atype
            atk_idx  = np.where(atk_mask)[0]
            n_t      = len(atk_idx)

            # threshold-free metrics — full normal pool
            sub_idx  = np.concatenate([atk_idx, normal_idx])
            y_sub    = y_labels[sub_idx]
            err_sub  = errors[sub_idx]
            auroc_t  = (roc_auc_score(y_sub, err_sub)
                        if y_sub.sum() > 0 and (y_sub == 0).any() else float("nan"))
            prauc_t  = (average_precision_score(y_sub, err_sub)
                        if y_sub.sum() > 0 and (y_sub == 0).any() else float("nan"))

            # threshold-based metrics — balanced 1:1 subset
            bal_normal  = rng.choice(normal_idx, size=min(n_t, len(normal_idx)), replace=False)
            bal_idx     = np.concatenate([atk_idx, bal_normal])
            y_bal       = y_labels[bal_idx]
            pred_bal    = y_pred[bal_idx]
            precision_t = precision_score(y_bal, pred_bal, zero_division=0)
            rec_t       = recall_score(y_bal, pred_bal, zero_division=0)
            acc_t       = accuracy_score(y_bal, pred_bal)
            f1_t        = f1_score(y_bal, pred_bal, zero_division=0)

            tp_t    = int(np.sum(y_pred[atk_idx] == 1))
            fn_t    = int(np.sum(y_pred[atk_idx] == 0))
            med_mse = float(np.median(errors[atk_mask]))

            per_type_lines.append(
                f"  {atype:<40} {auroc_t:>7.4f} {prauc_t:>7.4f} "
                f"{precision_t:>10.4f} {rec_t:>7.4f} {acc_t:>9.4f} {f1_t:>8.4f} {med_mse:>14.4f} {tp_t:>7} {fn_t:>7} {n_t:>7}"
            )
        per_type_lines.append(
            "  * Precision, Acc(bal), and F1(bal) computed on a 1:1 balanced subset (n attack rows : n sampled normal rows)"
        )

    # --- Save report ---
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        cm_block = (
            f"  TN={cm[0,0]}  FP={cm[0,1]}\n"
            f"  FN={cm[1,0]}  TP={cm[1,1]}"
        )
        lines = [
            "=== Global Generalisation Evaluation ===",
            f"Architecture:  {model_type}",
            f"Threshold:     {threshold:.4f}",
            f"AUROC:         {auroc:.4f}",
            f"PR-AUC:        {prauc:.4f}",
            f"Precision:     {precision:.4f}",
            f"Recall:        {recall_val:.4f}",
            f"Acc(bal):      {bal_acc:.4f}",
            f"F1(bal):       {f1_bal_g:.4f}",
            f"Normal Median MSE:    {normal_mse:.6f}",
            f"Attack Median MSE:    {attack_mse:.6f}",
            f"TP:            {tp_g}",
            f"FN:            {fn_g}",
            f"n:             {n_g}",
            "",
            "Confusion Matrix:",
            "  (rows=actual, cols=predicted)",
            cm_block,
        ]
        if per_type_lines:
            lines += ["", "Per-Attack-Type Metrics:"] + per_type_lines

        report_path = os.path.join(results_dir, f"{model_type}_global_report.txt")
        with open(report_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[Global Eval] Report saved → {report_path}")

def _aggregate_device_metrics(device_metrics: list[dict], results_dir, model_type: str) -> dict:
    """Aggregate local device metrics into network-level metrics.

    Threshold-based metrics are computed from the sum of all device confusion
    matrices. AUROC, PR-AUC, threshold, normal_mse and attack_mse are reported
    as sample-weighted averages using each device test size as weight.
    """

    if not device_metrics:
        print("[Network Metrics] No device metrics available.")
        return {}

    # Sum confusion matrices across devices
    total_cm = np.sum(
        [m["confusion_matrix"] for m in device_metrics],
        axis=0
    )

    tn, fp, fn, tp = total_cm.ravel()
    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (recall + specificity) / 2 if total > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Number of evaluated samples per device
    weights = np.array(
        [np.sum(m["confusion_matrix"]) for m in device_metrics],
        dtype=float
    )

    def weighted_metric(key: str) -> float:
        values = np.array([m[key] for m in device_metrics], dtype=float)
        mask = ~np.isnan(values)

        if not mask.any():
            return float("nan")

        return float(np.average(values[mask], weights=weights[mask]))

    aggregated = {
        "num_devices": len(device_metrics),
        "num_samples": int(total),
        "threshold_weighted": weighted_metric("threshold"),
        "auroc_weighted": weighted_metric("auroc"),
        "prauc_weighted": weighted_metric("prauc"),
        "accuracy": float(accuracy),
        "bal_acc": float(bal_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "normal_mse_weighted": weighted_metric("normal_mse"),
        "attack_mse_weighted": weighted_metric("attack_mse"),
        "confusion_matrix": total_cm,
    }

    print("\n=== Network-Level Device Metrics ===")
    print(
        f"Devices: {aggregated['num_devices']} | "
        f"Samples: {aggregated['num_samples']} | "
        f"AUROC(w): {aggregated['auroc_weighted']:.4f} | "
        f"PR-AUC(w): {aggregated['prauc_weighted']:.4f} | "
        f"Accuracy: {aggregated['accuracy']:.4f} | "
        f"Bal-Acc: {aggregated['bal_acc']:.4f} | "
        f"Precision: {aggregated['precision']:.4f} | "
        f"Recall: {aggregated['recall']:.4f} | "
        f"F1: {aggregated['f1']:.4f}"
    )

    print("Confusion matrix:")
    print(total_cm)

    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        report_path = os.path.join(results_dir, f"{model_type}_network_report.txt")

        lines = [
            "=== Network-Level Device Metrics ===",
            f"Architecture:              {model_type}",
            f"Devices:                   {aggregated['num_devices']}",
            f"Total samples:             {aggregated['num_samples']}",
            f"Weighted Threshold:        {aggregated['threshold_weighted']:.6f}",
            f"Weighted AUROC:            {aggregated['auroc_weighted']:.4f}",
            f"Weighted PR-AUC:           {aggregated['prauc_weighted']:.4f}",
            f"Accuracy:                  {aggregated['accuracy']:.4f}",
            f"Balanced Accuracy:         {aggregated['bal_acc']:.4f}",
            f"Precision:                 {aggregated['precision']:.4f}",
            f"Recall:                    {aggregated['recall']:.4f}",
            f"F1 Score:                  {aggregated['f1']:.4f}",
            f"Weighted Normal Median MSE:{aggregated['normal_mse_weighted']:.6f}",
            f"Weighted Attack Median MSE:{aggregated['attack_mse_weighted']:.6f}",
            "",
            "Confusion Matrix:",
            "  (rows=actual, cols=predicted)",
            f"  TN={tn}  FP={fp}",
            f"  FN={fn}  TP={tp}",
        ]

        with open(report_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"[Network Metrics] Report saved → {report_path}")

    return aggregated

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
    6. Evaluate the final global model on the shared general test set via
       :func:`_evaluate_global_test`, then save the model to ``src/models/``.
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
            local_epochs             = LOCAL_EPOCHS,
            early_stopping_patience  = EARLY_STOPPING_PATIENCE,
            gpu_lock                 = gpu_lock,
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
    
    device_metrics = [
        device.metrics
        for device in devices
        if device.metrics is not None
    ]
    
    _aggregate_device_metrics(
        device_metrics=device_metrics,
        results_dir=str(RESULTS_DIR),
        model_type=MODEL_TYPE,
    )

    # --- Step 6: global generalisation evaluation ---
    _evaluate_global_test(
        global_weights    = server.global_model,
        all_train_paths   = all_train_paths,
        all_val_paths     = [p["val"] for p in all_paths],
        known_categories  = known_categories,
        model_type        = MODEL_TYPE,
        input_dim         = input_dim,
        window_size       = WINDOW_SIZE,
        general_test_path = str(GENERAL_TEST_PATH),
        results_dir       = str(RESULTS_DIR),
        gpu_lock          = gpu_lock,
    )

    # Save the final global model to src/models/
    server.save_global_model(MODELS_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
