import threading

class CentralServer:
    def __init__(self, total_clients, rounds_to_simulate):
        self.global_model = 0.0
        self.updates = []
        self.lock = threading.Lock()
        self.total_clients = total_clients
        self.rounds_to_simulate = rounds_to_simulate
        self.current_round = 1
        
        # MAGIA: La barrera espera a 'total_clients' hilos.
        # Cuando llega el último, ejecuta 'aggregate_and_update' antes de soltarlos.
        self.sync_barrier = threading.Barrier(self.total_clients, action=self.aggregate_and_update)

    def receive_update(self, client_id, local_update):
        with self.lock:
            self.updates.append(local_update)
            print(f"[Servidor] Recibida actualización del Dispositivo {client_id}. ({len(self.updates)}/{self.total_clients})")
        
        # Pausa el hilo de este dispositivo hasta que lleguen los demás.
        self.sync_barrier.wait()

    def aggregate_and_update(self):
        # Esta función es llamada automáticamente por la Barrera
        print(f"\n--- Agregando modelos de la Ronda {self.current_round} ---")
        self.global_model = sum(self.updates) / len(self.updates)
        self.updates = [] # Limpiamos para la siguiente ronda
        
        print(f"[Servidor] Nuevo modelo global actualizado: {self.global_model:.4f}\n")
        
        self.current_round += 1
        if self.current_round > self.rounds_to_simulate:
            print("[Servidor] Simulación de Aprendizaje Federado completada.")