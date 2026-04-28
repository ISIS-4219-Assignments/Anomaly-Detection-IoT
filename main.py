# Importamos las clases desde nuestros módulos locales
from server import CentralServer
from device import SimulatedDevice

def main():
    NUM_CLIENTS = 4
    NUM_ROUNDS = 3

    # 1. Instanciar el servidor
    server = CentralServer(total_clients=NUM_CLIENTS, rounds_to_simulate=NUM_ROUNDS)

    devices = []
    
    # 2. Crear clientes con diferentes perfiles de velocidad/conexión
    # Dispositivos rápidos y estables (Media de 1s, desviación baja)
    devices.append(SimulatedDevice(client_id=1, server=server, dist_type='gauss', mu=1, sigma=0.2))
    devices.append(SimulatedDevice(client_id=2, server=server, dist_type='gauss', mu=1.5, sigma=0.3))
    
    # Dispositivo lento (Media de 4s)
    devices.append(SimulatedDevice(client_id=3, server=server, dist_type='gauss', mu=4, sigma=0.5))
    
    # Dispositivo impredecible (Distribución exponencial, media de 3s)
    devices.append(SimulatedDevice(client_id=4, server=server, dist_type='expo', mu=3))

    print("--- Iniciando Simulación de Aprendizaje Federado ---\n")
    
    # 3. Iniciar todos los hilos
    for device in devices:
        device.start()

    # 4. Esperar a que todos los dispositivos terminen su ejecución
    for device in devices:
        device.join()

    print("\nTodos los hilos cerrados. Fin del script.")

if __name__ == "__main__":
    main()