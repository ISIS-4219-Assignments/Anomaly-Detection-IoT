import threading
import time
import random

class SimulatedDevice(threading.Thread):
    def __init__(self, client_id, server, dist_type, mu, sigma=None):
        super().__init__()
        self.client_id = client_id
        self.server = server
        self.dist_type = dist_type
        self.mu = mu
        self.sigma = sigma

    def run(self):
        while self.server.current_round <= self.server.rounds_to_simulate:
            
            # 1. Descargar modelo global
            local_model = self.server.global_model
            
            # 2. Simular tiempo de entrenamiento
            if self.dist_type == 'gauss':
                delay = max(0.1, random.gauss(self.mu, self.sigma))
            elif self.dist_type == 'expo':
                delay = random.expovariate(1.0 / self.mu)
            
            print(f"[Dispositivo {self.client_id}] Entrenando... (Tomará {delay:.2f}s)")
            time.sleep(delay)
            
            # 3. Mejora del modelo local
            improvement = random.uniform(0.5, 2.0) 
            local_update = local_model + improvement
            
            # 4. Enviar al servidor
            # El dispositivo se quedará pausado DENTRO de esta función por la barrera
            # hasta que la ronda acabe. ¡Cero riesgos de desincronización!
            self.server.receive_update(self.client_id, local_update)