# Deployment — FL IoT Anomaly Detection

Arquitectura backend + frontend para entrenar y desplegar el modelo de detección de anomalías federado.

## Estructura

```
deployment/
├── backend/
│   ├── api.py               # FastAPI — endpoints REST + WebSocket
│   ├── fl_hooks.py          # HookedDevice + HookedServer (subclases sin modificar el repo)
│   ├── training_runner.py   # TrainingRunner: bridge sync training → async WS
│   └── requirements.txt
├── frontend/
│   ├── app.py               # Streamlit dashboard
│   ├── network_graph.py     # Visualización Plotly de la red FL
│   └── requirements.txt
└── README.md
```

## Prerequisitos

- Python 3.11+
- Dataset EdgeIIoTSet descargado y splits generados (ver `src/utils/`)
- Los 4 modelos `.keras` en `src/models/` (ya incluidos en el repo)

## Setup

### Backend

```bash
cd deployment/backend
pip install -r requirements.txt
```

### Frontend

```bash
cd deployment/frontend
pip install -r requirements.txt
```

## Ejecutar

### Terminal 1 — Backend

```bash
cd deployment/backend
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

El backend quedará disponible en `http://localhost:8000`.

Documentación automática: `http://localhost:8000/docs`

### Terminal 2 — Frontend

```bash
cd deployment/frontend
streamlit run app.py
```

Abrir en el navegador: `http://localhost:8501`

## Uso

### Tab: Entrenamiento

1. Configurar el backend URL en el sidebar (default: `http://localhost:8000`)
2. Seleccionar modelo, rondas y épocas
3. Click en **Iniciar Entrenamiento**
4. Observar la animación en tiempo real:
   - **Naranja** — dispositivo entrenando localmente
   - **Azul** — dispositivo enviando pesos al servidor (paquete animado)
   - **Morado** — servidor ejecutando FedAvg
   - **Verde** — dispositivo completado

### Tab: Inferencia

1. El backend debe estar corriendo
2. Subir un archivo CSV con tráfico de red (mismo formato que el dataset)
3. Click en **Detectar anomalías**
4. Ver errores de reconstrucción, umbral y clasificación por muestra

> **Nota:** La inferencia requiere que el preprocesador global esté guardado.
> Esto ocurre automáticamente al finalizar un entrenamiento con `/train`.
> Si usas el modelo pre-entrenado sin entrenar, ejecuta primero:
> `GET http://localhost:8000/status` para verificar el estado.

## Endpoints API

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/status` | Estado actual del entrenamiento |
| `POST` | `/train` | Iniciar nueva ronda de entrenamiento FL |
| `WS` | `/ws/training` | Stream de eventos en tiempo real |
| `POST` | `/predict` | Inferencia sobre CSV subido |

### Esquema de eventos WebSocket

```json
{"type": "setup_complete",    "input_dim": 63, "model_type": "conv1d", "num_rounds": 7, "devices": [...]}
{"type": "device_ready",      "device": "Distance", "train_samples": 4521}
{"type": "device_training",   "device": "Distance", "round": 1}
{"type": "epoch_end",         "device": "Distance", "round": 1, "epoch": 5, "train_loss": 0.004, "val_loss": 0.003}
{"type": "device_uploading",  "device": "Distance", "round": 1, "val_loss": 0.003}
{"type": "server_aggregating","round": 1}
{"type": "round_done",        "round": 1, "total_rounds": 7}
{"type": "device_done",       "device": "Distance", "metrics": {"auroc": 0.999, "recall": 0.995, "f1_bal": 0.997}}
{"type": "training_complete"}
```

## Despliegue en Streamlit Community Cloud

El frontend puede desplegarse en Streamlit Cloud apuntando a un backend accesible públicamente:

1. Hacer fork / push del repo a GitHub
2. En Streamlit Cloud, seleccionar `deployment/frontend/app.py` como entry point
3. En el sidebar de la app, actualizar el **Backend URL** con la URL pública del servidor

> El backend requiere acceso al dataset. Para Streamlit Cloud solo (sin backend propio),
> la tab de Inferencia funciona si se adapta para cargar el modelo directamente
> desde `src/models/conv1d_autoencoder_final.keras` (ya incluido en el repo).

## Modelos disponibles

| Modelo | AUROC | Recall | F1 (bal) | Archivo |
|--------|-------|--------|----------|---------|
| **Conv1D** | 0.9999 | 0.9995 | 0.9986 | `src/models/conv1d_autoencoder_final.keras` |
| LSTM | 0.9994 | 0.9958 | 0.9961 | `src/models/lstm_autoencoder_final.keras` |
| Transformer | 0.9978 | 0.9448 | 0.9716 | `src/models/transformer_autoencoder_final.keras` |
| Vanilla | 0.9962 | 0.9411 | 0.9634 | `src/models/vanilla_autoencoder_final.keras` |

El modelo recomendado es **Conv1D**: detecta los 14 tipos de ataque con recall perfecto.
