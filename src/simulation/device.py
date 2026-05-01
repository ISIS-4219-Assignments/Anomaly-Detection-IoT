"""
device.py
---------

Simulated edge device for the Federated Learning simulation.

Each device is a thread that runs the full local training loop:

1. Load and preprocess its private data split once (before the round loop).
2. For every round:
   a. Build a fresh Keras model and load the current global weights.
   b. Train locally for ``local_epochs`` epochs (autoencoder: target == input).
   c. Send the updated weights to the server and wait at the barrier.

The device never sees other devices' data — only its own train/val splits.
All inter-device communication goes through :class:`~server.CentralServer`.
"""


from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from keras.callbacks import EarlyStopping
from preprocessor import IoTPreprocessor
from models.factory import build_model
from windowing import create_windows
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for threads
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import threading
import os


class SimulatedDevice(threading.Thread):

    """A thread that simulates a federated learning client device.

    Each instance loads its own private data split, preprocesses it, and
    trains a local model for every communication round.  After training it
    submits its weights to the server via :meth:`~server.CentralServer.receive_update`
    and blocks at the synchronisation barrier until all other devices are done.

    Attributes
    ----------
    client_id : int or str
        Unique identifier used in log messages.
    server : CentralServer
        Reference to the central server.
    data_paths : dict[str, str]
        Paths to the device's data splits.  Expected keys: ``"train"``,
        ``"val"``, ``"test"``.
    model_type : str
        Architecture name — one of ``"vanilla"``, ``"lstm"``, ``"conv1d"``.
    input_dim : int
        Number of features after preprocessing; determines model input size.
    window_size : int or None
        Window length for sequence models; ``None`` for ``"vanilla"``.
    known_categories : dict[str, list] or None
        Global category vocabulary from :func:`~preprocessor.build_global_categories`.
        Passed to :class:`~preprocessor.IoTPreprocessor` to guarantee a
        uniform feature space across all devices.
    local_epochs : int
        Number of epochs to train locally each round.
    """

    def __init__(
        self,
        client_id: int | str,
        server,
        data_paths: dict[str, str],
        model_type: str,
        input_dim: int,
        window_size: int | None = None,
        known_categories: dict | None = None,
        local_epochs: int = 5,
        early_stopping_patience: int = 3,
        gpu_lock: threading.Lock | None = None,
        results_dir: str | None = None,
    ):
        
        """Initialise the device thread.

        Parameters
        ----------
        client_id : int or str
            Unique identifier for this device.
        server : CentralServer
            The central server managing the global model and synchronisation.
        data_paths : dict[str, str]
            Mapping of split name to CSV file path.  Must contain at least
            ``"train"`` and ``"val"`` keys.
        model_type : str
            Architecture to train.  One of ``"vanilla"``, ``"lstm"``,
            ``"conv1d"``.
        input_dim : int
            Feature dimension after preprocessing.
        window_size : int or None
            Sliding-window size used by :func:`~windowing.create_windows`.
            Must match the value used when computing ``input_dim``.
            Pass ``None`` for the vanilla model.
        known_categories : dict[str, list] or None
            Global vocabulary for categorical one-hot encoding.  If ``None``
            the preprocessor falls back to local-only categories, which may
            produce a different feature space across devices.
        local_epochs : int, optional
            How many epochs to train in each federated round.  Default: 5.
        early_stopping_patience : int, optional
            Stop a local training round early if val_loss does not improve by
            at least 1e-4 for this many consecutive epochs.  Weights are
            restored to the best epoch before the update is sent to the server.
            Set to 0 to disable early stopping.  Default: 3.
        """

        super().__init__()
        self.client_id = client_id
        self.server = server
        self.data_paths = data_paths
        self.model_type = model_type
        self.input_dim = input_dim
        self.window_size = window_size
        self.known_categories = known_categories
        self.local_epochs = local_epochs
        self.early_stopping_patience = early_stopping_patience
        self.gpu_lock = gpu_lock or threading.Lock()
        self.results_dir = results_dir
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------


    def _load_and_prepare(self) -> tuple[np.ndarray, np.ndarray]:

        """Load CSV splits, preprocess, and apply windowing if needed.

        The scaler is fitted on the training split and reused for validation,
        which is the correct procedure in both standard ML and federated
        learning.

        Returns
        -------
        X_train : np.ndarray
            Prepared training array.
            Shape ``(N_train, input_dim)`` for vanilla;
            ``(N_train - W + 1, W, input_dim)`` for sequence models.
        X_val : np.ndarray
            Prepared validation array (same shape convention as X_train).
        """

        pre = IoTPreprocessor(known_categories = self.known_categories)

        train_df = pd.read_csv(self.data_paths["train"], low_memory = False)
        val_df = pd.read_csv(self.data_paths["val"], low_memory = False)

        X_train, _ = pre.fit_transform(train_df)
        X_val, _ = pre.transform(val_df)

        # Keep the fitted preprocessor so evaluate() can reuse the scaler
        self._preprocessor = pre

        X_train = X_train.values.astype("float32")
        X_val = X_val.values.astype("float32")

        if self.window_size is not None:
            X_train = create_windows(X_train, self.window_size)
            X_val = create_windows(X_val, self.window_size)

        return X_train, X_val


    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------


    def run(self) -> None:

        """Execute the device's federated learning loop.

        Data is loaded once before the round loop to avoid redundant I/O.
        For each round the device:

        1. Builds a local model and loads the current global weights.
        2. Trains for ``local_epochs`` epochs (autoencoder loss: MSE of
           reconstruction).
        3. Calls :meth:`~server.CentralServer.receive_update` to submit
           weights and synchronise with the barrier.
        """

        print(f"[Device {self.client_id}] Loading and preprocessing data...")
        X_train, X_val = self._load_and_prepare()
        print(
            f"[Device {self.client_id}] Ready. "
            f"train={X_train.shape}, val={X_val.shape}"
        )

        while self.server.current_round <= self.server.rounds_to_simulate:
            round_num = self.server.current_round
            print(
                f"[Device {self.client_id}] Round {round_num} — "
                f"building model and loading global weights..."
            )

            # Build a fresh model and load the current global weights
            model = build_model(self.model_type, self.input_dim, self.window_size)
            model.set_weights(self.server.global_model)

            # Train locally; autoencoder target is the input itself
            callbacks = []
            if self.early_stopping_patience > 0:
                callbacks.append(EarlyStopping(
                    monitor            = "val_loss",
                    patience           = self.early_stopping_patience,
                    min_delta          = 1e-4,
                    restore_best_weights = True,
                    verbose            = 0,
                ))

            with self.gpu_lock:
                history = model.fit(
                    X_train,
                    X_train,
                    validation_data = (X_val, X_val),
                    epochs          = self.local_epochs,
                    batch_size      = 64,
                    callbacks       = callbacks,
                    verbose         = 0,
                )

            self._train_losses.extend(history.history["loss"])
            self._val_losses.extend(history.history["val_loss"])

            train_loss = history.history["loss"][-1]
            val_loss   = history.history["val_loss"][-1]
            print(
                f"[Device {self.client_id}] Round {round_num} — "
                f"loss: {train_loss:.4f}, val_loss: {val_loss:.4f}. "
                "Sending weights to server..."
            )

                # Submit weights and block at the barrier until all devices finish
            self.server.receive_update(self.client_id, model.get_weights())

        # ---- Threshold phase ----
        print(f"[Device {self.client_id}] Computing anomaly threshold...")

        final_model = build_model(self.model_type, self.input_dim, self.window_size)
        final_model.set_weights(self.server.global_model)

        with self.gpu_lock:
            X_val_pred = final_model.predict(X_val, batch_size = 256, verbose = 0)

        axes = tuple(range(1, X_val.ndim))
        val_errors = np.mean((X_val - X_val_pred) ** 2, axis = axes)

        threshold = float(np.percentile(val_errors, 99))

        print(
            f"[Device {self.client_id}] Threshold computed from validation: "
            f"{threshold:.6f}"
        )
        # ---- Evaluation phase (runs after all training rounds complete) ----
        metrics = self.evaluate(self.server.global_model, threshold)
        self._save_results(metrics)


    def evaluate(self, global_weights: list[np.ndarray], threshold: float) -> None:

        """Evaluate the global model on this device's test split.

        Loads the test CSV, applies the fitted scaler from training (no
        re-fitting), optionally windows the data, then computes per-sample
        reconstruction error.  Because the test split contains both normal
        and attack traffic, the method reports:

        - **normal_mse** — mean reconstruction error on normal samples.
        - **attack_mse** — mean reconstruction error on attack samples.
          A well-trained anomaly detector produces a clearly higher value
          here than ``normal_mse``.
        - **AUROC** — area under the ROC curve using reconstruction error
          as the anomaly score.  Threshold-free; 1.0 is perfect, 0.5 is
          random.

        For windowed models the label assigned to each window is the label
        of its *last* row (most recent trace in the window).

        Parameters
        ----------
        global_weights : list[np.ndarray]
            Final global model weights from the server, in Keras
            ``get_weights()`` format.
        """

        print(f"[Device {self.client_id}] Running test evaluation...")

        # Reuse the scaler fitted during training — never refit on test data
        test_df = pd.read_csv(self.data_paths["test"], low_memory = False)
        X_test, y_test = self._preprocessor.transform(test_df)

        X_test = X_test.values.astype("float32")
        y_test = y_test["Attack_label"].values

        if self.window_size is not None:
            X_test = create_windows(X_test, self.window_size)
            # Label of a window = label of its last row
            y_test = y_test[self.window_size - 1:]

        # Build model and load final global weights
        model = build_model(self.model_type, self.input_dim, self.window_size)
        model.set_weights(global_weights)

        with self.gpu_lock:
            X_pred = model.predict(X_test, batch_size = 256, verbose = 0)

        # Per-sample reconstruction error (MSE averaged over all non-batch dims)
        axes = tuple(range(1, X_test.ndim))  # (1,) for vanilla; (1, 2) for sequence
        errors = np.mean((X_test - X_pred) ** 2, axis = axes)
        
        y_pred = (errors > threshold).astype(int)

        normal_mask = y_test == 0
        attack_mask = y_test == 1

        normal_mse = float(np.median(errors[normal_mask])) if normal_mask.any() else float("nan")
        attack_mse = float(np.median(errors[attack_mask])) if attack_mask.any() else float("nan")
        precision = precision_score(y_test, y_pred, zero_division = 0)
        recall = recall_score(y_test, y_pred, zero_division = 0)
        f1 = f1_score(y_test, y_pred, zero_division = 0)
        cm = confusion_matrix(y_test, y_pred)

        if normal_mask.any() and attack_mask.any():
            auroc = roc_auc_score(y_test, errors)
        else:
            auroc = float("nan")

        print(
            f"[Device {self.client_id}] Test results — "
            f"Threshold: {threshold:.6f} | "
            f"AUROC: {auroc:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f} | "
            f"normal_median_mse: {normal_mse:.6f} | "
            f"attack_median_mse: {attack_mse:.6f}"
        )

        print(f"[Device {self.client_id}] Confusion matrix:")
        print(cm)

        return {
            "threshold":       threshold,
            "auroc":           auroc,
            "precision":       precision,
            "recall":          recall,
            "f1":              f1,
            "normal_mse":      normal_mse,
            "attack_mse":      attack_mse,
            "confusion_matrix": cm,
        }


    def _save_results(self, metrics: dict) -> None:

        if self.results_dir is None:
            return

        out_dir = os.path.join(self.results_dir, str(self.client_id))
        os.makedirs(out_dir, exist_ok=True)
        prefix = os.path.join(out_dir, self.model_type)

        # --- loss curve ---
        epochs = range(1, len(self._train_losses) + 1)
        fig, ax = plt.subplots()
        ax.plot(epochs, self._train_losses, label="train loss")
        ax.plot(epochs, self._val_losses,   label="val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title(f"Device {self.client_id} — {self.model_type} — Loss Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{prefix}_loss_curve.png", dpi=150)
        plt.close(fig)

        # --- text report ---
        cm = metrics["confusion_matrix"]
        rounds = self.server.rounds_to_simulate
        report = (
            f"Device:        {self.client_id}\n"
            f"Architecture:  {self.model_type}\n"
            f"Rounds:        {rounds}\n"
            f"Local epochs:  {self.local_epochs} (max, early stopping patience={self.early_stopping_patience})\n"
            f"Total epochs:  {len(self._train_losses)}\n"
            f"\n"
            f"=== Evaluation Results ===\n"
            f"Threshold:         {metrics['threshold']:.6f}\n"
            f"AUROC:             {metrics['auroc']:.4f}\n"
            f"Precision:         {metrics['precision']:.4f}\n"
            f"Recall:            {metrics['recall']:.4f}\n"
            f"F1 Score:          {metrics['f1']:.4f}\n"
            f"Normal Median MSE: {metrics['normal_mse']:.6f}\n"
            f"Attack Median MSE: {metrics['attack_mse']:.6f}\n"
            f"\n"
            f"Confusion Matrix:\n"
            f"  (rows=actual, cols=predicted)\n"
            f"  TN={cm[0,0]}  FP={cm[0,1]}\n"
            f"  FN={cm[1,0]}  TP={cm[1,1]}\n"
        )
        with open(f"{prefix}_report.txt", "w") as f:
            f.write(report)

        print(f"[Device {self.client_id}] Results saved → {out_dir}/")
