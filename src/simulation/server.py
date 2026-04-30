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


class CentralServer:

    """Orchestrates the Federated Learning process and maintains global weights.

    Attributes
    ----------
    global_model : list[np.ndarray]
        Current global model weights in Keras ``get_weights()`` format.
        Initialised from a freshly-built model so the weight structure is
        always consistent with the chosen architecture.
    updates : list[list[np.ndarray]]
        Local weight updates collected during the current round.  Cleared
        after every aggregation.
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
            One of ``"vanilla"``, ``"lstm"``, ``"conv1d"``.
        input_dim : int
            Number of features after preprocessing.  Passed to the model
            builder to determine layer sizes.
        window_size : int or None
            Sliding-window length; required for ``"lstm"`` and ``"conv1d"``,
            ``None`` for ``"vanilla"``.
        """

        # Build a throw-away model to capture the initial weight structure
        initial_model = build_model(model_type, input_dim, window_size)
        self.global_model: list[np.ndarray] = initial_model.get_weights()

        self.updates: list[list[np.ndarray]] = []
        self.lock = threading.Lock()
        self.total_clients = total_clients
        self.rounds_to_simulate = rounds_to_simulate
        self.current_round = 1

        # When the last thread arrives the barrier fires aggregate_and_update
        # before releasing all threads, so no device ever sees a partial update.
        self.sync_barrier = threading.Barrier(
            self.total_clients, action=self.aggregate_and_update
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def receive_update(
        self, client_id: int | str, local_weights: list[np.ndarray]
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
        """

        with self.lock:
            self.updates.append(local_weights)
            print(
                f"[Server] Received update from Device {client_id}. "
                f"({len(self.updates)}/{self.total_clients})"
            )

        # Block until all clients reach this point; last one aggregates
        self.sync_barrier.wait()

    # ------------------------------------------------------------------
    # Internal – called automatically by the barrier
    # ------------------------------------------------------------------

    def aggregate_and_update(self) -> None:

        """Average all collected local weights and update the global model.

        Implements FedAvg: for each weight tensor, the new global value is
        the element-wise mean of the corresponding tensors from all clients.
        The updates list is cleared and the round counter incremented before
        threads are released.

        This method is invoked automatically by :attr:`sync_barrier` and
        should never be called directly.
        """
        
        print(f"\n--- Aggregating models for Round {self.current_round} ---")

        # FedAvg: layer-wise mean across all client weight lists
        self.global_model = [
            np.mean([client_weights[i] for client_weights in self.updates], axis=0)
            for i in range(len(self.global_model))
        ]

        self.updates = []  # clear for next round
        print(f"[Server] Global model updated after round {self.current_round}.\n")

        self.current_round += 1

        if self.current_round > self.rounds_to_simulate:
            print("[Server] Federated Learning simulation completed.")
