"""app.py

Live network-attack simulation dashboard for the FL IoT Anomaly Detection system.

Instead of requiring a CSV upload, the dashboard auto-loads attack traffic from
the repository's ``data/attack_traffic/`` directory and immediately runs the
Conv1D autoencoder on a selectable scenario.  The network animation is the hero
element; device status cards and a classification log follow below it.

Inference pipeline
------------------
1. Load scenario CSVs  →  IoTPreprocessor.transform() (or fit_transform if no cache)
2. create_windows(window_size=30) per device
3. model.predict()  →  per-window median squared error
4. threshold = 99th-percentile of reconstruction errors
5. anomaly = mse > threshold
6. Build interleaved timeline (round-robin across devices)
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
_DATA_DIR = _PROJECT_ROOT / "data" / "attack_traffic"
_WINDOW_SIZE = 30

sys.path.insert(0, str(_SIM_DIR))

_SCENARIOS = {
    "DDoS Multi-Vector": {
        "description": "Cuatro dispositivos bajo diferentes vectores de ataque DDoS",
        "devices": [
            {"name": "Sensor_Distancia", "icon": "📏", "csv": "DDoS_UDP_Flood_attack.csv",  "rows": 80},
            {"name": "Sensor_Llama",     "icon": "🔥", "csv": "DDoS_HTTP_Flood_attack.csv", "rows": 80},
            {"name": "Receptor_IR",      "icon": "📡", "csv": "DDoS_TCP_SYN_Flood_attack.csv", "rows": 80},
            {"name": "Sensor_Sonido",    "icon": "🔊", "csv": "DDoS_ICMP_Flood_attack.csv", "rows": 80},
        ],
    },
    "Ataques Mixtos": {
        "description": "Cinco dispositivos IoT bajo distintos tipos de ciberataque",
        "devices": [
            {"name": "Termostato",   "icon": "🌡️", "csv": "Port_Scanning_attack.csv",  "rows": 80},
            {"name": "Cerradura",    "icon": "🔐", "csv": "Backdoor_attack.csv",        "rows": 80},
            {"name": "Camara_IP",    "icon": "📷", "csv": "XSS_attack.csv",             "rows": 80},
            {"name": "Control_HVAC", "icon": "❄️", "csv": "SQL_injection_attack.csv",  "rows": 80},
            {"name": "Medidor",      "icon": "⚡", "csv": "Ransomware_attack.csv",      "rows": 80},
        ],
    },
    "MITM y Reconocimiento": {
        "description": "Ataques de reconocimiento, MITM y escaneo de vulnerabilidades",
        "devices": [
            {"name": "Gateway_IoT",  "icon": "🔗", "csv": "MITM_attack.csv",                     "rows": 80},
            {"name": "Router",       "icon": "📶", "csv": "OS_Fingerprinting_attack.csv",         "rows": 80},
            {"name": "Hub_Central",  "icon": "🖧",  "csv": "Vulnerability_scanner_attack.csv",    "rows": 80},
            {"name": "Broker_MQTT",  "icon": "📨", "csv": "Uploading_attack.csv",                 "rows": 80},
        ],
    },
}

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


def _compute_scenario(scenario_name: str, model) -> dict:
    """Load CSVs, run inference, and build the interleaved window timeline.

    Parameters
    ----------
    scenario_name : str
        Key into ``_SCENARIOS``.
    model : keras.Model
        Loaded Conv1D autoencoder.

    Returns
    -------
    dict
        Keys: ``devices`` (per-device info dicts with errors/anomaly arrays),
        ``all_windows`` (list of dicts for the classification log),
        ``all_errors`` (flat array), ``all_anomaly`` (flat array),
        ``window_devs`` (flat list of device names, one per window).
        Returns ``{}`` if no devices could be loaded.
    """
    from windowing import create_windows
    from preprocessor import IoTPreprocessor

    scenario = _SCENARIOS[scenario_name]
    preprocessor = _load_cached_preprocessor()

    device_frames: list[pd.DataFrame] = []
    valid_devices: list[dict] = []

    for dev in scenario["devices"]:
        csv_path = _DATA_DIR / dev["csv"]
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, nrows=dev["rows"], low_memory=False)
            device_frames.append(df)
            valid_devices.append(dev)
        except Exception:
            continue

    if not valid_devices:
        return {}

    if preprocessor is not None:
        processed_arrays: list[np.ndarray] = []
        for df in device_frames:
            try:
                X, _ = preprocessor.transform(df)
                processed_arrays.append(X.values.astype("float32"))
            except Exception:
                processed_arrays.append(None)
    else:
        combined = pd.concat(device_frames, ignore_index=True)
        pre = IoTPreprocessor(known_categories=None)
        X_combined, _ = pre.fit_transform(combined)
        X_combined_arr = X_combined.values.astype("float32")
        processed_arrays = []
        cursor = 0
        for df in device_frames:
            n = len(df)
            processed_arrays.append(X_combined_arr[cursor: cursor + n])
            cursor += n

    devices_info: list[dict] = []
    for dev, X_dev in zip(valid_devices, processed_arrays):
        if X_dev is None:
            continue
        try:
            X_windows = create_windows(X_dev, _WINDOW_SIZE)
            if X_windows.shape[0] == 0:
                continue
            errors, threshold = _run_inference(X_windows, model)
            anomaly = (errors > threshold).astype(int)
            devices_info.append({
                "name": dev["name"],
                "icon": dev["icon"],
                "csv": dev["csv"],
                "errors": errors,
                "anomaly": anomaly,
                "threshold": threshold,
                "n_windows": len(errors),
                "n_anomaly": int(anomaly.sum()),
                "attack_label": _ATTACK_LABELS.get(dev["csv"], dev["csv"]),
            })
        except Exception:
            continue

    if not devices_info:
        return {}

    max_windows = max(d["n_windows"] for d in devices_info)
    all_windows: list[dict] = []
    all_errors_list: list[float] = []
    all_anomaly_list: list[int] = []
    window_devs: list[str] = []

    for idx in range(max_windows):
        for dev in devices_info:
            if idx < dev["n_windows"]:
                all_windows.append({
                    "dispositivo": dev["name"],
                    "ataque": dev["attack_label"],
                    "mse": round(float(dev["errors"][idx]), 8),
                    "clasificacion": "🔴 Ataque" if dev["anomaly"][idx] == 1 else "🟢 Normal",
                })
                all_errors_list.append(float(dev["errors"][idx]))
                all_anomaly_list.append(int(dev["anomaly"][idx]))
                window_devs.append(dev["name"])

    return {
        "devices": devices_info,
        "all_windows": all_windows,
        "all_errors": np.array(all_errors_list, dtype=float),
        "all_anomaly": np.array(all_anomaly_list, dtype=int),
        "window_devs": window_devs,
    }


def _render_device_card(col, name: str, info: dict) -> None:
    """Render an HTML status card for a single IoT device into a Streamlit column.

    Parameters
    ----------
    col : streamlit.delta_generator.DeltaGenerator
        The column object to render into.
    name : str
        Device name.
    info : dict
        Device info dict from ``_compute_scenario`` (must have icon, n_anomaly,
        n_windows, attack_label keys).
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
    padding: 16px 12px;
    box-shadow: {glow};
    text-align: center;
    margin-bottom: 8px;
">
    <div style="font-size: 2rem; margin-bottom: 6px;">{info['icon']}</div>
    <div style="color: #ecf0f1; font-weight: 600; font-size: 0.85rem;
                margin-bottom: 8px; word-break: break-word;">
        {name.replace('_', ' ')}
    </div>
    <div style="
        display: inline-block;
        background: {status_bg};
        color: white;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 20px;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    ">{status_text}</div>
    <div style="color: #bdc3c7; font-size: 0.75rem; margin-bottom: 4px;">
        {info['attack_label']}
    </div>
    <div style="color: #e74c3c; font-weight: 700; font-size: 1.1rem;">
        {n_anomaly}
        <span style="color: #7f8c8d; font-weight: 400; font-size: 0.8rem;">
            / {n_windows}
        </span>
    </div>
    <div style="color: #95a5a6; font-size: 0.72rem;">
        {pct:.1f}% anomalías
    </div>
</div>
"""
    col.markdown(html, unsafe_allow_html=True)


def main() -> None:
    """Entry point for the live network-attack simulation Streamlit dashboard."""
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

        st.markdown("### Escenario de simulación")
        selected_scenario = st.radio(
            label="Selecciona escenario",
            options=list(_SCENARIOS.keys()),
            label_visibility="collapsed",
        )
        st.caption(_SCENARIOS[selected_scenario]["description"])

        st.divider()

        with st.expander("📂 Datos personalizados"):
            st.caption(
                "Sube un CSV en formato EdgeIIoTSet para ejecutar la inferencia "
                f"directamente (mín. {_WINDOW_SIZE + 1} filas)."
            )
            uploaded = st.file_uploader("Selecciona un CSV de tráfico de red", type=["csv"])
            if uploaded is not None:
                with st.spinner("Ejecutando inferencia sobre archivo personalizado…"):
                    try:
                        from windowing import create_windows
                        from preprocessor import IoTPreprocessor

                        df_up = pd.read_csv(uploaded, low_memory=False)
                        if len(df_up) < _WINDOW_SIZE + 1:
                            st.error(
                                f"El archivo tiene {len(df_up)} filas — "
                                f"se necesitan al menos {_WINDOW_SIZE + 1}."
                            )
                        else:
                            preprocessor = _load_cached_preprocessor()
                            if preprocessor is not None:
                                X_up, _ = preprocessor.transform(df_up)
                            else:
                                pre = IoTPreprocessor(known_categories=None)
                                X_up, _ = pre.fit_transform(df_up)
                            X_win_up = create_windows(X_up.values.astype("float32"), _WINDOW_SIZE)
                            errs_up, thr_up = _run_inference(X_win_up, model)
                            anom_up = (errs_up > thr_up).astype(int)
                            st.metric("Ventanas",   len(errs_up))
                            st.metric("Anomalías",  int(anom_up.sum()))
                            st.metric("Umbral MSE", f"{thr_up:.6f}")
                    except Exception as exc:
                        st.error(f"Error: {exc}")

    st.title("🔒 IoT Network Attack Monitor")
    st.caption(
        "Simulación en vivo de ataques IoT con Conv1D autoencoder federado — "
        "EdgeIIoTSet dataset · 14 tipos de ataque"
    )

    cache_key = f"scenario_{selected_scenario}"
    if cache_key not in st.session_state:
        with st.spinner(f"⚙️ Procesando escenario '{selected_scenario}'…"):
            st.session_state[cache_key] = _compute_scenario(selected_scenario, model)

    data = st.session_state[cache_key]

    if not data:
        st.error(
            "No se pudieron cargar datos para este escenario. "
            "Verifica que los archivos CSV existen en `data/attack_traffic/`."
        )
        return

    devices_info = data["devices"]
    all_errors   = data["all_errors"]
    all_anomaly  = data["all_anomaly"]
    window_devs  = data["window_devs"]
    all_windows  = data["all_windows"]

    total_windows   = len(all_anomaly)
    total_attacks   = int(all_anomaly.sum())
    devs_under_attack = sum(1 for d in devices_info if d["n_anomaly"] > 0)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Ventanas analizadas",     total_windows)
    col_m2.metric("Ataques detectados",      total_attacks)
    col_m3.metric("Dispositivos bajo ataque", devs_under_attack)

    st.markdown("---")

    from attack_animation import build_attack_animation

    device_names_list = [d["name"] for d in devices_info]
    device_attack_types = {d["name"]: d["attack_label"] for d in devices_info}

    st.plotly_chart(
        build_attack_animation(
            device_names=device_names_list,
            anomaly=all_anomaly,
            errors=all_errors,
            window_devs=window_devs,
            device_attack_types=device_attack_types,
        ),
        use_container_width=True,
    )

    st.markdown("#### Estado de Dispositivos")

    n_cols = min(len(devices_info), 5)
    card_cols = st.columns(n_cols)
    for idx, dev in enumerate(devices_info):
        _render_device_card(card_cols[idx % n_cols], dev["name"], dev)

    st.markdown("---")

    st.markdown("#### Registro de Clasificación")
    df_log = pd.DataFrame(all_windows)
    st.dataframe(
        df_log,
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
