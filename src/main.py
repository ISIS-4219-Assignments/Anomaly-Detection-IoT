# Import classes from our local modules
from device import SimulatedDevice
from server import CentralServer

def main():

    """
    The main entry point for the Federated Learning simulation.

    This function sets up the simulation environment by initializing the 
    central server and creating multiple simulated client devices. It assigns 
    different statistical profiles to the devices to mimic real-world edge 
    environments (e.g., fast connections, slow connections, and unpredictable 
    network drops). Finally, it starts all threads and waits for the simulation 
    to finish.
    """

    NUM_CLIENTS = 4
    NUM_ROUNDS = 3

    # 1. Instantiate the central server
    server = CentralServer(total_clients = NUM_CLIENTS, rounds_to_simulate = NUM_ROUNDS)

    devices = []
    
    # 2. Create clients with different speed/connection profiles
    
    # Fast and stable devices (Mean of 1s or 1.5s, low standard deviation)
    devices.append(SimulatedDevice(client_id = 1, server = server, dist_type = 'gauss', mu = 1, sigma = 0.2))
    devices.append(SimulatedDevice(client_id = 2, server = server, dist_type = 'gauss', mu = 1.5, sigma = 0.3))
    
    # Slow device (Mean of 4s)
    devices.append(SimulatedDevice(client_id = 3, server = server, dist_type = 'gauss', mu = 4, sigma = 0.5))
    
    # Unpredictable device (Exponential distribution, mean of 3s)
    devices.append(SimulatedDevice(client_id = 4, server = server, dist_type = 'expo', mu = 3))

    print("--- Starting Federated Learning Simulation ---\n")
    
    # 3. Start all device threads
    for device in devices:
        device.start()

    # 4. Wait for all devices to finish their execution (block main thread until done)
    for device in devices:
        device.join()

    print("\nAll threads closed. End of script.")

if __name__ == "__main__":
    main()