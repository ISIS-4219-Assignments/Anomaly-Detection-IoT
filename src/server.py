import threading


class CentralServer:

    """
    Orchestrates the Federated Learning process and maintains the global model.
    
    This class is responsible for collecting local model updates from all 
    connected client devices in a thread-safe manner. Once all clients have 
    submitted their updates for the current round, it aggregates them 
    (simulating Federated Averaging or FedAvg) and starts the next round.

    Attributes:

        global_model (float): The current weights of the global model (simplified as a float).
        updates (list): A list temporarily storing the local updates received in the current round.
        lock (threading.Lock): A mutual exclusion lock to prevent race conditions when appending updates.
        total_clients (int): The total number of client devices participating in the training.
        rounds_to_simulate (int): The total number of federated learning rounds to execute.
        current_round (int): Tracks the current training round.
        sync_barrier (threading.Barrier): A synchronization primitive that blocks client threads
            until all updates are received, then triggers the aggregation automatically.
    """

    def __init__(self, total_clients, rounds_to_simulate):

        """
        Initializes the CentralServer instance.
        
        Args:

            total_clients (int): The number of devices that will participate in each round.
            rounds_to_simulate (int): How many communication rounds to perform before stopping.
        """

        self.global_model = 0.0  # Simulated model weights
        self.updates = []
        self.lock = threading.Lock() # Crucial to prevent threads from colliding
        self.total_clients = total_clients
        self.rounds_to_simulate = rounds_to_simulate
        self.current_round = 1
        
        # The barrier waits for exactly 'total_clients' threads.
        # When the very last thread arrives, it automatically executes the 'action' 
        # (aggregate_and_update) before releasing all threads to start the next round.
        self.sync_barrier = threading.Barrier(self.total_clients, action=self.aggregate_and_update)

    def receive_update(self, client_id, local_update):

        """
        Receives and stores a local model update from a client device.
        
        This method uses a threading lock to ensure that multiple devices 
        can send their updates concurrently without corrupting the updates list.
        After saving the update, it pauses the calling thread until all other 
        devices finish their training for the current round.

        Args:
            client_id (int or str): The identifier of the device sending the update.
            local_update (float): The updated model weights from the device.
        """

        # Acquire lock to safely append to the shared list
        with self.lock:
            self.updates.append(local_update)
            print(f"[Server] Received update from Device {client_id}. ({len(self.updates)}/{self.total_clients})")
        
        # Pause this device's thread until all other devices reach this exact point
        self.sync_barrier.wait()

    def aggregate_and_update(self):

        """
        Aggregates the collected local updates and updates the global model.
        
        This function is automatically triggered by the threading.Barrier once 
        all clients have submitted their updates. It calculates the average of 
        the received weights (simulating FedAvg), clears the update list for 
        the next round, and increments the round counter.
        """
        
        print(f"\n--- Aggregating models for Round {self.current_round} ---")
        
        # FedAvg: We average the weights (simulated here with simple mathematical averages)
        self.global_model = sum(self.updates) / len(self.updates)
        self.updates = [] # Clear the list for the next round
        
        print(f"[Server] New global model updated: {self.global_model:.4f}\n")
        
        self.current_round += 1
        
        # Check if the simulation has finished
        if self.current_round > self.rounds_to_simulate:
            print("[Server] Federated Learning simulation completed.")