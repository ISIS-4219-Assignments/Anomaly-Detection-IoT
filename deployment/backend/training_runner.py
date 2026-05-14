"""training_runner.py

TrainingRunner service: orchestrates FL training in a background thread and
fans out events to async WebSocket subscribers (Observer pattern).

Sync-to-async bridge
--------------------
Keras and the threading primitives in SimulatedDevice are synchronous.
FastAPI's WebSocket layer is async.  TrainingRunner decouples them:

  training thread ──(queue.Queue)──► _drain_to_subscribers() ──► asyncio.Queue per WS client

The drain coroutine runs inside FastAPI's asyncio event loop and wakes up
every 50 ms to flush the sync queue into every subscriber's async queue.
"""

import asyncio
import pickle
import queue
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SIM_DIR = _PROJECT_ROOT / "src" / "simulation"
sys.path.insert(0, str(_SIM_DIR))

from preprocessor import IoTPreprocessor, build_global_categories
from fl_hooks import HookedDevice, HookedServer

_SPLITS_DIR = _PROJECT_ROOT / "data" / "splits"
_MODELS_DIR = _PROJECT_ROOT / "src" / "models"
_RESULTS_DIR = _PROJECT_ROOT / "results"
PREPROCESSOR_CACHE_PATH = Path(__file__).resolve().parent / "preprocessor_cache.pkl"

DEVICE_NAMES: list[str] = [
    "Distance",
    "Flame_Sensor",
    "IR_Receiver",
    "phValue",
    "Soil_Moisture",
    "Sound_Sensor",
    "Temperature_and_Humidity",
    "Water_Level",
]

MODEL_WINDOW_SIZES: dict[str, int | None] = {
    "vanilla": None,
    "lstm": 30,
    "conv1d": 30,
    "transformer": 30,
}


@dataclass
class TrainingConfig:
    """Hyperparameters and device selection for a single FL run."""

    model_type: str = "conv1d"
    num_rounds: int = 7
    local_epochs: int = 20
    early_stopping_patience: int = 3
    device_names: list[str] = field(default_factory=lambda: list(DEVICE_NAMES))

    def window_size(self) -> int | None:
        """Return the appropriate window size for the chosen model_type."""
        return MODEL_WINDOW_SIZES.get(self.model_type)


class TrainingRunner:
    """Orchestrates FL training and bridges sync events to async subscribers.

    Design pattern: Observer.
    - Subjects emit events through a sync queue (training thread).
    - _drain_to_subscribers() relays them to each subscriber's asyncio.Queue.
    - Consumers call subscribe() to receive a dedicated async queue.
    """

    def __init__(self) -> None:
        """Initialize with idle status and no active training thread."""
        self._sync_queue: queue.Queue = queue.Queue()
        self._subscribers: list[asyncio.Queue] = []
        self._sub_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.status: str = "idle"
        self.current_round: int = 0
        self.config: TrainingConfig | None = None

    def start(self, config: TrainingConfig, loop: asyncio.AbstractEventLoop) -> None:
        """Launch FL training in a background daemon thread.

        Parameters
        ----------
        config : TrainingConfig
            Hyperparameters and device list for this run.
        loop : asyncio.AbstractEventLoop
            The running event loop; used to schedule the drain coroutine.
        """
        self.config = config
        self.status = "training"
        self.current_round = 1
        self._sync_queue = queue.Queue()

        self._thread = threading.Thread(
            target=self._run_in_thread,
            args=(config,),
            daemon=True,
        )
        self._thread.start()

        asyncio.run_coroutine_threadsafe(self._drain_to_subscribers(), loop)

    def subscribe(self) -> asyncio.Queue:
        """Register a new async subscriber and return its dedicated event queue.

        Returns
        -------
        asyncio.Queue
            The caller must await get() on this queue to receive events.
        """
        q: asyncio.Queue = asyncio.Queue()
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Deregister a subscriber queue."""
        with self._sub_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    @property
    def is_running(self) -> bool:
        """True while a training run is in progress."""
        return self.status == "training"

    async def _drain_to_subscribers(self) -> None:
        """Drain sync_queue and fan out each event to all async subscribers."""
        while True:
            try:
                event = self._sync_queue.get_nowait()

                if event.get("type") == "round_done":
                    self.current_round = int(event.get("round", self.current_round)) + 1

                if event.get("type") == "training_complete":
                    self.status = "done"

                with self._sub_lock:
                    targets = list(self._subscribers)
                for q in targets:
                    await q.put(event)

                if event.get("type") == "training_complete":
                    break

            except queue.Empty:
                await asyncio.sleep(0.05)
            except Exception:
                await asyncio.sleep(0.1)

    def _run_in_thread(self, config: TrainingConfig) -> None:
        """Training thread entry point; always emits training_complete on exit."""
        try:
            self._execute_simulation(config)
        finally:
            self._sync_queue.put({"type": "training_complete"})
            self.status = "done"

    def _make_device_paths(self, name: str) -> dict[str, str]:
        """Build the train/val/test path mapping for a device directory."""
        base = _SPLITS_DIR / name
        return {
            "train": str(base / "train.csv"),
            "val": str(base / "val.csv"),
            "test": str(base / "test.csv"),
        }

    def _compute_input_dim(self, train_path: str, known_categories: dict) -> int:
        """Derive the feature count by preprocessing one training split."""
        pre = IoTPreprocessor(known_categories=known_categories)
        X, _ = pre.fit_transform(pd.read_csv(train_path, low_memory=False))
        return int(X.shape[1])

    def _execute_simulation(self, config: TrainingConfig) -> None:
        """Build HookedServer + HookedDevices and run the full FL simulation."""
        window_size = config.window_size()
        all_paths = [self._make_device_paths(name) for name in config.device_names]
        all_train_paths = [p["train"] for p in all_paths]

        self._sync_queue.put({"type": "setup_started"})

        known_categories = build_global_categories(all_train_paths)
        input_dim = self._compute_input_dim(all_train_paths[0], known_categories)

        self._sync_queue.put({
            "type": "setup_complete",
            "input_dim": input_dim,
            "model_type": config.model_type,
            "num_rounds": config.num_rounds,
            "devices": config.device_names,
        })

        server = HookedServer(
            total_clients=len(config.device_names),
            rounds_to_simulate=config.num_rounds,
            model_type=config.model_type,
            input_dim=input_dim,
            window_size=window_size,
            event_queue=self._sync_queue,
        )

        gpu_lock = threading.Lock()
        devices = [
            HookedDevice(
                client_id=name,
                server=server,
                data_paths=paths,
                model_type=config.model_type,
                input_dim=input_dim,
                window_size=window_size,
                known_categories=known_categories,
                local_epochs=config.local_epochs,
                early_stopping_patience=config.early_stopping_patience,
                gpu_lock=gpu_lock,
                results_dir=str(_RESULTS_DIR),
                event_queue=self._sync_queue,
            )
            for name, paths in zip(config.device_names, all_paths)
        ]

        for device in devices:
            device.start()
        for device in devices:
            device.join()

        server.save_global_model(str(_MODELS_DIR))
        self._persist_preprocessor(known_categories, all_train_paths)

    def _persist_preprocessor(
        self, known_categories: dict, train_paths: list[str]
    ) -> None:
        """Fit and save a global preprocessor for use in the inference endpoint."""
        pooled = pd.concat(
            [pd.read_csv(p, low_memory=False) for p in train_paths],
            ignore_index=True,
        )
        pre = IoTPreprocessor(known_categories=known_categories)
        pre.fit_transform(pooled)

        PREPROCESSOR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PREPROCESSOR_CACHE_PATH, "wb") as f:
            pickle.dump({"preprocessor": pre, "known_categories": known_categories}, f)
