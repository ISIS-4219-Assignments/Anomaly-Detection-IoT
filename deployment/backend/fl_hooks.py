"""fl_hooks.py

Instrumented subclasses of SimulatedDevice and CentralServer that emit
structured lifecycle events to a queue.  No original source files are modified.

HookedDevice overrides run() to:
  - inject an epoch-level Keras callback (per-epoch loss events)
  - emit device_uploading before submitting weights to the server

HookedServer overrides aggregate_and_update() to bracket the parent call
with server_aggregating and round_done events.
"""

import queue
import sys
import numpy as np
from pathlib import Path
from keras.callbacks import EarlyStopping, Callback

_SIM_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "simulation"
sys.path.insert(0, str(_SIM_DIR))

from device import SimulatedDevice
from server import CentralServer
from models.factory import build_model


class _EpochEventCallback(Callback):
    """Keras callback that pushes per-epoch metrics to the event queue."""

    def __init__(
        self,
        device_id: str,
        round_num: int,
        event_queue: queue.Queue,
    ) -> None:
        """
        Parameters
        ----------
        device_id : str
            Device identifier included in every event payload.
        round_num : int
            Current federated round number.
        event_queue : queue.Queue
            Destination queue for emitted events.
        """
        super().__init__()
        self._device_id = device_id
        self._round_num = round_num
        self._queue = event_queue

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Emit an epoch_end event after each training epoch."""
        logs = logs or {}
        self._queue.put({
            "type": "epoch_end",
            "device": self._device_id,
            "round": self._round_num,
            "epoch": epoch + 1,
            "train_loss": float(logs.get("loss", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)),
        })


class HookedDevice(SimulatedDevice):
    """SimulatedDevice that emits structured lifecycle events during FL training.

    Extends SimulatedDevice by accepting an optional event_queue.  When provided,
    training milestones are pushed as dicts with a 'type' key.  When omitted,
    behaviour is identical to the parent class.

    The run() override is the minimal surface needed: it replicates the parent's
    training loop and adds three injection points that Python inheritance cannot
    expose without a full method override.
    """

    def __init__(
        self,
        *args,
        event_queue: queue.Queue | None = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        event_queue : queue.Queue or None
            Queue that receives lifecycle events.  Pass None to disable emission.
        All remaining positional and keyword arguments are forwarded to
        SimulatedDevice.__init__.
        """
        super().__init__(*args, **kwargs)
        self._event_queue = event_queue

    def _emit(self, event: dict) -> None:
        """Push event to the queue if one is attached."""
        if self._event_queue is not None:
            self._event_queue.put(event)

    def run(self) -> None:
        """Execute the FL training loop with event emission.

        Replicates SimulatedDevice.run() and adds:
          1. device_ready after data is loaded
          2. device_training at the start of each round
          3. _EpochEventCallback injected into model.fit() callbacks
          4. device_uploading before weight submission to the server
          5. device_done after evaluation completes
        """
        X_train, X_val = self._load_and_prepare()

        self._emit({
            "type": "device_ready",
            "device": str(self.client_id),
            "train_samples": int(X_train.shape[0]),
        })

        while self.server.current_round <= self.server.rounds_to_simulate:
            round_num = self.server.current_round

            self._emit({
                "type": "device_training",
                "device": str(self.client_id),
                "round": round_num,
            })

            model = build_model(self.model_type, self.input_dim, self.window_size)
            model.set_weights(self.server.global_model)

            callbacks: list = []
            if self._event_queue is not None:
                callbacks.append(
                    _EpochEventCallback(str(self.client_id), round_num, self._event_queue)
                )
            if self.early_stopping_patience > 0:
                callbacks.append(EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    min_delta=1e-4,
                    restore_best_weights=True,
                    verbose=0,
                ))

            with self.gpu_lock:
                history = model.fit(
                    X_train, X_train,
                    validation_data=(X_val, X_val),
                    epochs=self.local_epochs,
                    batch_size=64,
                    callbacks=callbacks,
                    verbose=0,
                )

            self._train_losses.extend(history.history["loss"])
            self._val_losses.extend(history.history["val_loss"])

            self._emit({
                "type": "device_uploading",
                "device": str(self.client_id),
                "round": round_num,
                "val_loss": float(history.history["val_loss"][-1]),
            })

            self.server.receive_update(
                self.client_id,
                model.get_weights(),
                self._n_train_samples,
            )

        final_model = build_model(self.model_type, self.input_dim, self.window_size)
        final_model.set_weights(self.server.global_model)

        with self.gpu_lock:
            X_val_pred = final_model.predict(X_val, batch_size=256, verbose=0)

        axes = tuple(range(1, X_val.ndim))
        val_errors = np.median((X_val - X_val_pred) ** 2, axis=axes)
        threshold = float(np.percentile(val_errors, 99))

        metrics = self.evaluate(self.server.global_model, threshold)
        self.metrics = metrics
        self._save_results(metrics)

        auroc = metrics.get("auroc", float("nan"))
        f1_bal = metrics.get("f1_bal", float("nan"))

        self._emit({
            "type": "device_done",
            "device": str(self.client_id),
            "metrics": {
                "auroc": float(auroc) if not np.isnan(auroc) else None,
                "recall": float(metrics.get("recall", 0.0)),
                "f1_bal": float(f1_bal) if not np.isnan(f1_bal) else None,
            },
        })


class HookedServer(CentralServer):
    """CentralServer that emits aggregation lifecycle events to a queue."""

    def __init__(
        self,
        *args,
        event_queue: queue.Queue | None = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        event_queue : queue.Queue or None
            Queue that receives server-side events.
        All remaining arguments are forwarded to CentralServer.__init__.
        """
        super().__init__(*args, **kwargs)
        self._event_queue = event_queue

    def aggregate_and_update(self) -> None:
        """Bracket FedAvg aggregation with server_aggregating and round_done events."""
        round_num = self.current_round

        if self._event_queue is not None:
            self._event_queue.put({
                "type": "server_aggregating",
                "round": round_num,
            })

        super().aggregate_and_update()

        if self._event_queue is not None:
            self._event_queue.put({
                "type": "round_done",
                "round": round_num,
                "total_rounds": self.rounds_to_simulate,
            })
