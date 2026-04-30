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


from preprocessor import IoTPreprocessor
from models.factory import build_model
from windowing import create_windows
import pandas as pd
import numpy as np
import threading
from sklearn.metrics import roc_auc_score


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

        pre = IoTPreprocessor(known_categories=self.known_categories)

        train_df = pd.read_csv(self.data_paths["train"], low_memory=False)
        val_df = pd.read_csv(self.data_paths["val"], low_memory=False)

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
            history = model.fit(
                X_train,
                X_train,
                validation_data=(X_val, X_val),
                epochs=self.local_epochs,
                batch_size=64,
                verbose=0,
            )

            train_loss = history.history["loss"][-1]
            val_loss   = history.history["val_loss"][-1]
            print(
                f"[Device {self.client_id}] Round {round_num} — "
                f"loss: {train_loss:.4f}, val_loss: {val_loss:.4f}. "
                "Sending weights to server..."
            )

            # Submit weights and block at the barrier until all devices finish
            self.server.receive_update(self.client_id, model.get_weights())

        # ---- Evaluation phase (runs after all training rounds complete) ----
        self.evaluate(self.server.global_model)

    def evaluate(self, global_weights: list[np.ndarray]) -> None:
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
        test_df = pd.read_csv(self.data_paths["test"], low_memory=False)
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

        X_pred = model.predict(X_test, batch_size=256, verbose=0)

        # Per-sample reconstruction error (MSE averaged over all non-batch dims)
        axes = tuple(range(1, X_test.ndim))  # (1,) for vanilla; (1, 2) for sequence
        errors = np.mean((X_test - X_pred) ** 2, axis=axes)

        normal_mask = y_test == 0
        attack_mask = y_test == 1

        normal_mse = float(np.mean(errors[normal_mask])) if normal_mask.any() else float("nan")
        attack_mse = float(np.mean(errors[attack_mask])) if attack_mask.any() else float("nan")

        if normal_mask.any() and attack_mask.any():
            auroc = roc_auc_score(y_test, errors)
        else:
            auroc = float("nan")

        print(
            f"[Device {self.client_id}] Test results — "
            f"AUROC: {auroc:.4f} | "
            f"normal_mse: {normal_mse:.4f} | "
            f"attack_mse: {attack_mse:.4f}"
        )
