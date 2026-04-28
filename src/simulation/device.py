import threading
import random
import time


class SimulatedDevice(threading.Thread):

    """
    A thread class simulating a client device in a Federated Learning environment.
    
    This class mimics the behavior of an edge device that downloads a global model,
    simulates local training with a specific time delay based on a statistical 
    distribution, and then sends the updated local model back to the central server.

    Attributes:

        client_id (int or str): A unique identifier for the simulated device.
        server (CentralServer): The central server instance orchestrating the federated learning.
        dist_type (str): The statistical distribution used to simulate training delay ('gauss' or 'expo').
        mu (float): The mean value for the time delay distribution.
        sigma (float, optional): The standard deviation for the Gaussian distribution. Defaults to None.
    """


    def __init__(self, client_id, server, dist_type, mu, sigma = None):
        
        """
        Initializes the SimulatedDevice instance.
        
        Args:

            client_id (int or str): Unique ID for the device.
            server (CentralServer): Reference to the central server.
            dist_type (str): Type of distribution for the delay ('gauss' or 'expo').
            mu (float): Mean time (in seconds) for the training delay.
            sigma (float, optional): Standard deviation for the 'gauss' distribution. Defaults to None.
        """

        super().__init__()
        self.client_id = client_id
        self.server = server
        self.dist_type = dist_type
        self.mu = mu
        self.sigma = sigma


    def run(self):

        """
        The main execution loop for the thread.
        
        Loops through the specified number of training rounds. In each round, it:
        1. Retrieves the global model from the server.
        2. Simulates local training time using the configured statistical distribution.
        3. Updates the local model (simulating training on local data).
        4. Sends the update to the central server and waits for synchronization.
        """
        
        while self.server.current_round <= self.server.rounds_to_simulate:
            
            # 1. Download the global model
            local_model = self.server.global_model
            
            # 2. Simulate training time and network latency
            if self.dist_type == 'gauss':
                # Normal distribution (e.g., stable devices with minor variations)
                # Ensure the delay is at least 0.1 seconds to avoid negative times
                delay = max(0.1, random.gauss(self.mu, self.sigma))
            elif self.dist_type == 'expo':
                # Exponential distribution (e.g., highly variable waiting times, unstable networks)
                delay = random.expovariate(1.0 / self.mu)
            else:
                # Fallback delay if distribution type is unknown
                delay = self.mu
            
            print(f"[Device {self.client_id}] Training... (Will take {delay:.2f}s)")
            time.sleep(delay)
            
            # 3. Simulate local model improvement
            # In a real scenario, this is where `model.fit()` would happen
            improvement = random.uniform(0.5, 2.0) 
            local_update = local_model + improvement
            
            # 4. Send the updated model back to the server
            # The device will be paused inside this function by the server's threading.Barrier
            # until the round ends. This guarantees zero synchronization risks!
            self.server.receive_update(self.client_id, local_update)