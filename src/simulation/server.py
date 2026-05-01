"""
server.py
---------

Central server for the Federated Learning simulation.

The server owns the global model weights and coordinates all training rounds.
Weights are stored as a **list of NumPy arrays** — the format returned by
:meth:`keras.Model.get_weights` — so no PyTorch or framework-specific types
appear here.

Aggregation uses Federated Averaging (FedAvg): each layer's weights are
updated to the element-wise mean of the corresponding layers received from
all clients.

Synchronization design
----------------------
A :class:`threading.Barrier` is initialised with ``total_clients`` parties
and an ``action=self.aggregate_and_update`` callback.  When the last device
calls :meth:`receive_update`, the barrier automatically fires
``aggregate_and_update`` before releasing all threads into the next round.
This eliminates explicit round management in device threads and is
race-condition-free by construction.
"""


from models.factory import build_model
import numpy as np
import threading
import os


class CentralServer:

    """Orchestrates the Federated Learning process and maintains global weights.

    Attributes
    ----------
    global_model : list[np.ndarray]
        Current global model weights in Keras ``get_weights()`` format.
        Initialised from a freshly-built model so the weight structure is
        always consistent with the chosen architecture.
    updates : list[tuple[list[np.ndarray], int]]
        Local weight updates collected during the current round, stored as
        ``(weights, n_samples)`` tuples.  Cleared after every aggregation.
    lock : threading.Lock
        Guards appends to ``updates`` so concurrent device threads cannot
        corrupt the list.
    total_clients : int
        Number of devices participating in every round.
    rounds_to_simulate : int
        Total number of communication rounds to run.
    current_round : int
        1-based index of the round currently in progress.
    sync_barrier : threading.Barrier
        Blocks each device after it submits its update; released only when
        all ``total_clients`` devices have arrived.  Automatically triggers
        :meth:`aggregate_and_update` at that point.
    model_type : str
        Architecture name stored so :meth:`save_global_model` can rebuild
        the correct Keras model.
    input_dim : int
        Feature dimension stored for the same reason as ``model_type``.
    window_size : int or None
        Sliding-window length stored for the same reason as ``model_type``.
    """

    def __init__(
        self,
        total_clients: int,
        rounds_to_simulate: int,
        model_type: str,
        input_dim: int,
        window_size: int | None = None,
    ):
        
        """Initialise the server and the global model.

        A model is built once using :func:`~models.factory.build_model` solely
        to obtain the correct weight structure (shapes and dtypes).  The same
        structure is then expected from every device update throughout training.

        Parameters
        ----------
        total_clients : int
            Number of devices that will participate in each round.
        rounds_to_simulate : int
            How many communication rounds to run before stopping.
        model_type : str
            Architecture name forwarded to :func:`~models.factory.build_model`.
            One of ``"vanilla"``, ``"lstm"``, ``"conv1d"``, ``"transformer"``.
        input_dim : int
            Number of features after preprocessing.  Passed to the model
            builder to determine layer sizes.
        window_size : int or None
            Sliding-window length; required for ``"lstm"``, ``"conv1d"``, and
            ``"transformer"``; ``None`` for ``"vanilla"``.
        """

        # Build a throw-away model to capture the initial weight structure
        initial_model = build_model(model_type, input_dim, window_size)
        self.global_model: list[np.ndarray] = initial_model.get_weights()

        self.updates: list[tuple[list[np.ndarray], int]] = []
        self.lock = threading.Lock()
        self.total_clients = total_clients
        self.rounds_to_simulate = rounds_to_simulate
        self.current_round = 1

        # Stored so save_global_model() can rebuild the correct architecture
        self.model_type = model_type
        self.input_dim = input_dim
        self.window_size = window_size

        # When the last thread arrives the barrier fires aggregate_and_update
        # before releasing all threads, so no device ever sees a partial update.
        self.sync_barrier = threading.Barrier(
            self.total_clients, action = self.aggregate_and_update
        )


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------


    def receive_update(
        self, client_id: int | str, local_weights: list[np.ndarray], n_samples: int
    ) -> None:

        """Accept a local weight update and block until the round ends.

        The update is appended to :attr:`updates` under the lock to prevent
        concurrent writes from corrupting the list.  The thread then waits at
        the barrier; the last arriving thread triggers :meth:`aggregate_and_update`.

        Parameters
        ----------
        client_id : int or str
            Identifier of the submitting device (used for logging only).
        local_weights : list[np.ndarray]
            Weights from the device's locally-trained model in Keras
            ``get_weights()`` format.  Must match the structure of
            :attr:`global_model`.
        n_samples : int
            Number of training samples used by this device in the current round.
            Used to weight this device's contribution during aggregation so that
            larger devices have proportionally more influence (weighted FedAvg).
        """

        with self.lock:
            self.updates.append((local_weights, n_samples))
            print(
                f"[Server] Received update from Device {client_id} "
                f"(n={n_samples}). "
                f"({len(self.updates)}/{self.total_clients})"
            )

        # Block until all clients reach this point; last one aggregates
        self.sync_barrier.wait()


    # ------------------------------------------------------------------
    # Internal – called automatically by the barrier
    # ------------------------------------------------------------------


    def aggregate_and_update(self) -> None:

        """Aggregate collected local weights with sample-weighted FedAvg.

        Each device's contribution to the new global model is proportional to
        its training set size (``n_samples`` passed via :meth:`receive_update`),
        so devices with more data have more influence.  The updates list is
        cleared and the round counter incremented before threads are released.

        This method is invoked automatically by :attr:`sync_barrier` and
        should never be called directly.
        """
        
        print(f"\n--- Aggregating models for Round {self.current_round} ---")

        # Weighted FedAvg: each device's contribution is proportional to its
        # training set size so larger devices have more influence on the global model.
        weights_list = [w for w, _ in self.updates]
        counts       = np.array([n for _, n in self.updates], dtype=np.float64)
        total        = counts.sum()
        self.global_model = [
            np.sum([weights_list[j][i] * counts[j] for j in range(len(self.updates))], axis=0) / total
            for i in range(len(self.global_model))
        ]

        self.updates = []  # clear for next round
        print(f"[Server] Global model updated after round {self.current_round}.\n")

        self.current_round += 1

        if self.current_round > self.rounds_to_simulate:
            print("[Server] Federated Learning simulation completed.")


    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------


    def save_global_model(self, save_dir: str) -> str:

        """Build a Keras model, load the current global weights, and save it.

        The file is saved in Keras's native ``.keras`` format under
        ``save_dir/<model_type>_autoencoder_final.keras``.  The directory is
        created automatically if it does not exist.

        Parameters
        ----------
        save_dir : str
            Directory where the model file will be written.

        Returns
        -------
        str
            Absolute path of the saved model file.
        """

        model = build_model(self.model_type, self.input_dim, self.window_size)
        model.set_weights(self.global_model)

        os.makedirs(save_dir, exist_ok = True)
        path = os.path.join(save_dir, f"{self.model_type}_autoencoder_final.keras")
        model.save(path)
        print(f"[Server] Global model saved → {path}")
        return path
