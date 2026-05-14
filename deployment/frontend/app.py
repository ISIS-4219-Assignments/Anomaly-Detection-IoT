"""app.py

Streamlit inference dashboard for the FL IoT Anomaly Detection system.

Loads the pre-trained Conv1D autoencoder directly from the repository and
runs anomaly detection on CSV files uploaded by the user.  No backend server
is required.

Inference pipeline
------------------
1. Upload CSV  →  IoTPreprocessor.transform()
2. create_windows(window_size=30)
3. model.predict()  →  per-sample median squared error
4. threshold = 99th percentile of reconstruction errors
5. anomaly = mse > threshold
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SIM_DIR = _PROJECT_ROOT / "src" / "simulation"
_MODEL_PATH = _PROJECT_ROOT / "src" / "models" / "conv1d_autoencoder_final.keras"
_PREPROCESSOR_PATH = Path(__file__).resolve().parent.parent / "preprocessor_cache.pkl"
_WINDOW_SIZE = 30

sys.path.insert(0, str(_SIM_DIR))


@st.cache_resource(show_spinner="Cargando modelo Conv1D…")
def _load_model():
    """Load and cache the Conv1D autoencoder from the repository."""
    import keras
    return keras.models.load_model(str(_MODEL_PATH))


@st.cache_resource(show_spinner="Cargando preprocesador…")
def _load_cached_preprocessor():
    """Load the global IoTPreprocessor from the pickle cache, or None if absent."""
    if not _PREPROCESSOR_PATH.exists():
        return None
    with open(_PREPROCESSOR_PATH, "rb") as f:
        cache = pickle.load(f)
    return cache["preprocessor"]


def _preprocess(df: pd.DataFrame) -> np.ndarray:
    """Transform a raw CSV DataFrame into windowed model input.

    Uses the cached global preprocessor when available; otherwise fits a local
    preprocessor directly on the uploaded DataFrame (no known_categories).
    The latter works correctly as long as the CSV covers all categorical values
    present in the training set — i.e. any standard EdgeIIoTSet file.

    Parameters
    ----------
    df : pd.DataFrame
        Raw network traffic rows in EdgeIIoTSet format.

    Returns
    -------
    np.ndarray
        Shape (n_windows, window_size, n_features) ready for model.predict().
    """
    from windowing import create_windows
    from preprocessor import IoTPreprocessor

    preprocessor = _load_cached_preprocessor()

    if preprocessor is not None:
        X, _ = preprocessor.transform(df)
    else:
        pre = IoTPreprocessor(known_categories=None)
        X, _ = pre.fit_transform(df)

    X_arr = X.values.astype("float32")
    return create_windows(X_arr, _WINDOW_SIZE)


def _run_inference(X_windows: np.ndarray, model) -> tuple[np.ndarray, float]:
    """Run the autoencoder and compute per-window reconstruction error.

    Parameters
    ----------
    X_windows : np.ndarray
        Windowed input of shape (n, window_size, features).
    model : keras.Model
        Loaded Conv1D autoencoder.

    Returns
    -------
    errors : np.ndarray
        Per-window median squared error.
    threshold : float
        99th-percentile of errors used to classify anomalies.
    """
    X_pred = model.predict(X_windows, batch_size=256, verbose=0)
    errors = np.median((X_windows - X_pred) ** 2, axis=(1, 2)).astype(float)
    threshold = float(np.percentile(errors, 99))
    return errors, threshold


def _scatter_figure(errors: np.ndarray, anomaly: np.ndarray, threshold: float) -> go.Figure:
    """Build MSE scatter plot with anomaly colouring and threshold line."""
    colors = np.where(anomaly == 1, "#e74c3c", "#2ecc71")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=errors.tolist(),
        mode="markers",
        marker=dict(color=colors.tolist(), size=4, opacity=0.75),
        name="MSE por ventana",
    ))
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text=f"Umbral ({threshold:.5f})",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="Ventana",
        yaxis_title="Reconstruction MSE",
        height=320,
        margin=dict(l=50, r=20, t=20, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def _histogram_figure(errors: np.ndarray, threshold: float) -> go.Figure:
    """Build MSE histogram with threshold line."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=errors.tolist(),
        nbinsx=60,
        marker_color="#3498db",
        opacity=0.8,
        name="MSE",
    ))
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text="Umbral",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="MSE",
        yaxis_title="Frecuencia",
        height=240,
        margin=dict(l=50, r=20, t=20, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def main() -> None:
    """Entry point for the Streamlit inference dashboard."""
    st.set_page_config(
        page_title="IoT Anomaly Detection",
        page_icon="🔒",
        layout="wide",
    )

    st.title("🔒 IoT Anomaly Detection")
    st.caption(
        "Detección de anomalías de red con Conv1D autoencoder federado — "
        "EdgeIIoTSet dataset · 14 tipos de ataque"
    )

    model = _load_model()

    if not _PREPROCESSOR_PATH.exists():
        st.info(
            "No se encontró caché de preprocesador — se ajustará sobre el CSV subido. "
            "Para máxima precisión ejecuta `python deployment/generate_preprocessor.py` "
            "con los datos de entrenamiento y sube `deployment/preprocessor_cache.pkl`.",
            icon="ℹ️",
        )

    with st.sidebar:
        st.markdown("### Modelo")
        st.success("Conv1D Autoencoder")
        st.metric("AUROC", "0.9999")
        st.metric("Recall", "99.95 %")
        st.metric("F1 (bal)", "0.9986")
        st.caption("Entrenado con Federated Learning sobre 8 dispositivos IoT.")

        st.divider()
        st.markdown("### Umbral")
        st.caption(
            "Se calcula como el percentil 99 de los errores de reconstrucción "
            "del archivo subido.  Muestras por encima del umbral se clasifican "
            "como anomalías."
        )

    st.markdown("#### Subir archivo CSV")
    st.caption(
        "El archivo debe tener el mismo formato que el dataset EdgeIIoTSet "
        f"(al menos {_WINDOW_SIZE + 1} filas para poder crear ventanas)."
    )

    uploaded = st.file_uploader("Selecciona un CSV de tráfico de red", type=["csv"])

    if uploaded is None:
        st.info("Sube un archivo CSV para comenzar la clasificación.")
        return

    with st.spinner("Preprocesando y ejecutando inferencia…"):
        try:
            df = pd.read_csv(uploaded, low_memory=False)

            if len(df) < _WINDOW_SIZE + 1:
                st.error(
                    f"El archivo tiene solo {len(df)} filas. "
                    f"Se necesitan al menos {_WINDOW_SIZE + 1}."
                )
                return

            X_windows = _preprocess(df)
            errors, threshold = _run_inference(X_windows, model)
            anomaly = (errors > threshold).astype(int)

        except Exception as exc:
            st.error(f"Error durante la inferencia: {exc}")
            return

    n_anomalies = int(anomaly.sum())
    n_total = len(errors)
    pct = 100 * n_anomalies / n_total if n_total > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ventanas analizadas", n_total)
    col2.metric("Anomalías detectadas", n_anomalies)
    col3.metric("Tasa de anomalías", f"{pct:.1f} %")
    col4.metric("Umbral MSE", f"{threshold:.6f}")

    st.plotly_chart(_scatter_figure(errors, anomaly, threshold), use_container_width=True)

    col_hist, col_table = st.columns([2, 3])

    with col_hist:
        st.plotly_chart(_histogram_figure(errors, threshold), use_container_width=True)

    with col_table:
        df_result = pd.DataFrame({
            "ventana": range(n_total),
            "mse": np.round(errors, 8),
            "clasificacion": np.where(anomaly == 1, "🔴 Anomalía", "🟢 Normal"),
        })
        st.dataframe(
            df_result,
            use_container_width=True,
            height=240,
            hide_index=True,
        )

    st.divider()
    st.markdown("#### Simulación de red")
    st.caption(
        "Paquetes fluyendo desde Internet hacia los dispositivos IoT.  "
        "El color de los dispositivos se va enrojeciendo al acumular ataques.  "
        "Presiona ▶ Reproducir para iniciar."
    )

    from attack_animation import build_attack_animation

    _DISPLAY_DEVICES = [
        "Distance", "Flame_Sensor", "IR_Receiver", "Sound_Sensor", "Water_Level"
    ]
    st.plotly_chart(
        build_attack_animation(
            device_names=_DISPLAY_DEVICES,
            anomaly=anomaly,
            errors=errors,
        ),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
