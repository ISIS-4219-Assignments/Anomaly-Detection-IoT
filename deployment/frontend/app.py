"""app.py

Live network-attack simulation dashboard for the FL IoT Anomaly Detection system.

The user uploads a CSV in EdgeIIoTSet format.  The Conv1D autoencoder runs
inference and drives the animated network graph showing attack vs. normal
traffic windows in real time.

Inference pipeline
------------------
1. Upload CSV  →  IoTPreprocessor.transform() (or fit_transform if no cache)
2. create_windows(window_size=30)
3. model.predict()  →  per-window median squared error
4. threshold = 99th-percentile of reconstruction errors
5. anomaly = mse > threshold
6. Build Plotly animation
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SIM_DIR = _PROJECT_ROOT / "src" / "simulation"
_MODEL_PATH = _PROJECT_ROOT / "src" / "models" / "conv1d_autoencoder_final.keras"
_PREPROCESSOR_PATH = Path(__file__).resolve().parent.parent / "preprocessor_cache.pkl"
_WINDOW_SIZE = 30

sys.path.insert(0, str(_SIM_DIR))

_ATTACK_LABELS = {
    "DDoS_UDP_Flood_attack.csv":        "DDoS UDP Flood",
    "DDoS_HTTP_Flood_attack.csv":       "DDoS HTTP Flood",
    "DDoS_TCP_SYN_Flood_attack.csv":    "DDoS TCP SYN",
    "DDoS_ICMP_Flood_attack.csv":       "DDoS ICMP Flood",
    "Port_Scanning_attack.csv":         "Port Scanning",
    "Backdoor_attack.csv":              "Backdoor",
    "XSS_attack.csv":                   "XSS",
    "SQL_injection_attack.csv":         "SQL Injection",
    "Ransomware_attack.csv":            "Ransomware",
    "MITM_attack.csv":                  "MITM",
    "OS_Fingerprinting_attack.csv":     "OS Fingerprinting",
    "Vulnerability_scanner_attack.csv": "Vuln. Scanner",
    "Uploading_attack.csv":             "Malicious Upload",
    "Password_attack.csv":              "Password Attack",
}

_DEVICE_ICONS = ["📏", "🔥", "📡", "🔊", "🌡️", "🔐", "📷", "❄️", "⚡", "🔗"]


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
    with open(_PREPROCESSOR_PATH, "rb") as fh:
        cache = pickle.load(fh)
    return cache["preprocessor"]


def _run_inference(X_windows: np.ndarray, model) -> tuple[np.ndarray, float]:
    """Run the autoencoder and return per-window reconstruction errors and threshold.

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


def _process_uploaded_file(uploaded_file, model) -> dict | None:
    """Preprocess an uploaded CSV, run inference, and return result dict.

    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit uploaded file object.
    model : keras.Model
        Loaded Conv1D autoencoder.

    Returns
    -------
    dict or None
        Result dict on success, None on failure (errors displayed via st.error).
    """
    from windowing import create_windows
    from preprocessor import IoTPreprocessor

    try:
        df = pd.read_csv(uploaded_file, low_memory=False)
    except Exception as exc:
        st.error(f"No se pudo leer el CSV: {exc}")
        return None

    if len(df) < _WINDOW_SIZE + 1:
        st.error(
            f"El archivo tiene solo {len(df)} filas. "
            f"Se necesitan al menos {_WINDOW_SIZE + 1}."
        )
        return None

    preprocessor = _load_cached_preprocessor()
    try:
        if preprocessor is not None:
            X, _ = preprocessor.transform(df)
        else:
            pre = IoTPreprocessor(known_categories=None)
            X, _ = pre.fit_transform(df)
        X_arr = X.values.astype("float32")
    except Exception as exc:
        st.error(f"Error en preprocesamiento: {exc}")
        return None

    try:
        X_windows = create_windows(X_arr, _WINDOW_SIZE)
        if X_windows.shape[0] == 0:
            st.error("No se pudieron crear ventanas con estos datos.")
            return None
        errors, threshold = _run_inference(X_windows, model)
        anomaly = (errors > threshold).astype(int)
    except Exception as exc:
        st.error(f"Error en inferencia: {exc}")
        return None

    filename = getattr(uploaded_file, "name", "upload.csv")
    attack_label = _ATTACK_LABELS.get(filename, filename.replace(".csv", "").replace("_", " "))
    device_name = "Dispositivo_IoT"

    all_windows = [
        {
            "ventana": i + 1,
            "mse": round(float(errors[i]), 8),
            "clasificacion": "🔴 Ataque" if anomaly[i] == 1 else "🟢 Normal",
        }
        for i in range(len(errors))
    ]

    return {
        "device_name": device_name,
        "attack_label": attack_label,
        "errors": errors,
        "anomaly": anomaly,
        "threshold": threshold,
        "n_windows": len(errors),
        "n_anomaly": int(anomaly.sum()),
        "all_windows": all_windows,
    }


def _render_device_card(name: str, info: dict) -> None:
    """Render an HTML status card for the analyzed IoT device.

    Parameters
    ----------
    name : str
        Device name.
    info : dict
        Result dict from ``_process_uploaded_file``.
    """
    n_anomaly = info["n_anomaly"]
    n_windows = info["n_windows"]
    pct = 100.0 * n_anomaly / n_windows if n_windows > 0 else 0.0
    border_color = "#e74c3c" if n_anomaly > 0 else "#2ecc71"
    glow = f"0 0 12px {border_color}55"
    status_bg = "#e74c3c" if n_anomaly > 0 else "#2ecc71"
    status_text = "BAJO ATAQUE" if n_anomaly > 0 else "NORMAL"

    html = f"""
<div style="
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 2px solid {border_color};
    border-radius: 12px;
    padding: 16px 24px;
    box-shadow: {glow};
    text-align: center;
    max-width: 320px;
    margin: 0 auto 16px auto;
">
    <div style="font-size: 2.4rem; margin-bottom: 6px;">📡</div>
    <div style="color: #ecf0f1; font-weight: 600; font-size: 0.9rem; margin-bottom: 8px;">
        {name.replace('_', ' ')}
    </div>
    <div style="
        display: inline-block;
        background: {status_bg};
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
    ">{status_text}</div>
    <div style="color: #bdc3c7; font-size: 0.8rem; margin-bottom: 6px;">
        {info['attack_label']}
    </div>
    <div style="color: #e74c3c; font-weight: 700; font-size: 1.3rem;">
        {n_anomaly}
        <span style="color: #7f8c8d; font-weight: 400; font-size: 0.85rem;">
            / {n_windows} ventanas
        </span>
    </div>
    <div style="color: #95a5a6; font-size: 0.75rem;">{pct:.1f}% anomalías</div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def _upload_prompt() -> None:
    """Render the landing screen shown when no file has been uploaded."""
    st.markdown(
        """
<div style="
    text-align: center;
    padding: 60px 20px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    border: 1px solid #2c3e50;
    margin: 20px 0;
">
    <div style="font-size: 4rem; margin-bottom: 16px;">🔒</div>
    <h2 style="color: #ecf0f1; margin-bottom: 8px;">Sube un CSV de tráfico de red</h2>
    <p style="color: #95a5a6; font-size: 0.95rem; max-width: 480px; margin: 0 auto;">
        Usa el panel lateral para cargar un archivo <b style="color:#ecf0f1;">CSV del dataset EdgeIIoTSet</b>.
        El modelo Conv1D federado clasificará cada ventana de tráfico en
        <span style="color:#2ecc71;">Normal</span> o
        <span style="color:#e74c3c;">Ataque</span>
        y animará el flujo en tiempo real.
    </p>
    <div style="margin-top: 24px; color: #7f8c8d; font-size: 0.8rem;">
        Mínimo {min_rows} filas · Formato EdgeIIoTSet · 78 features
    </div>
</div>
""".replace("{min_rows}", str(_WINDOW_SIZE + 1)),
        unsafe_allow_html=True,
    )


def main() -> None:
    """Entry point for the IoT network-attack simulation Streamlit dashboard."""
    st.set_page_config(
        page_title="IoT Attack Monitor",
        page_icon="🔒",
        layout="wide",
    )

    model = _load_model()

    with st.sidebar:
        st.markdown("## 🔒 IoT Attack Monitor")
        st.divider()

        st.markdown("### Modelo")
        st.success("Conv1D Autoencoder — FL")
        st.metric("AUROC",    "0.9999")
        st.metric("Recall",   "99.95 %")
        st.metric("F1 (bal)", "0.9986")
        st.caption("Entrenado con Federated Learning sobre 8 dispositivos IoT.")

        st.divider()

        st.markdown("### 📂 Cargar datos")
        st.caption(
            f"Sube un CSV en formato EdgeIIoTSet "
            f"(mín. {_WINDOW_SIZE + 1} filas)."
        )
        uploaded = st.file_uploader(
            label="Selecciona un CSV de tráfico de red",
            type=["csv"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            if st.button("🔄 Limpiar y cargar otro", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key.startswith("result_"):
                        del st.session_state[key]
                st.rerun()

    st.title("🔒 IoT Network Attack Monitor")
    st.caption(
        "Detección de anomalías IoT con Conv1D autoencoder federado — "
        "EdgeIIoTSet dataset · 14 tipos de ataque"
    )

    if uploaded is None:
        _upload_prompt()
        return

    cache_key = f"result_{uploaded.name}_{uploaded.size}"
    if cache_key not in st.session_state:
        with st.spinner("⚙️ Ejecutando inferencia…"):
            st.session_state[cache_key] = _process_uploaded_file(uploaded, model)

    data = st.session_state[cache_key]

    if data is None:
        return

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Archivo",               uploaded.name)
    col_m2.metric("Ventanas analizadas",   data["n_windows"])
    col_m3.metric("Ataques detectados",    data["n_anomaly"])
    col_m4.metric("Umbral MSE",            f"{data['threshold']:.6f}")

    st.markdown("---")

    from attack_animation import build_attack_animation

    st.plotly_chart(
        build_attack_animation(
            device_names=[data["device_name"]],
            anomaly=data["anomaly"],
            errors=data["errors"],
        ),
        use_container_width=True,
    )

    st.markdown("#### Estado del Dispositivo")
    _, card_col, _ = st.columns([1, 2, 1])
    with card_col:
        _render_device_card(data["device_name"], data)

    st.markdown("---")
    st.markdown("#### Registro de Clasificación")
    st.dataframe(
        pd.DataFrame(data["all_windows"]),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
