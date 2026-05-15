"""app.py

Live network-attack simulation dashboard for the FL IoT Anomaly Detection system.

Upload one or more CSV files in EdgeIIoTSet format:
- 1 file  → single device, treated as one traffic stream.
- 2–5 files → one device per file; the animation shows all simultaneously.

Inference pipeline
------------------
1. Upload CSV(s)  →  IoTPreprocessor.transform() (or fit_transform if no cache)
2. create_windows(window_size=30) per device
3. model.predict()  →  per-window median squared error
4. threshold = 99th-percentile of reconstruction errors (computed per device)
5. anomaly = mse > threshold
6. Interleave windows across devices → build Plotly animation
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
_MAX_DEVICES = 14
_MAX_FILE_BYTES = 1 * 1024 * 1024 * 1024
_MAX_ANIMATION_WINDOWS = 300
_CHUNK_SIZE = 5_000
_LARGE_FILE_THRESHOLD = 50 * 1024 * 1024  # 50 MB

sys.path.insert(0, str(_SIM_DIR))

_DEVICE_NAMES = [
    "Termostato",    "Camara_IP",    "Sensor_Temp",  "Medidor_Luz",
    "Cerradura",     "Router_IoT",   "Sensor_Gas",   "Monitor_CO2",
    "Control_HVAC",  "Interruptor",  "Sensor_Mov",   "Gateway",
    "Broker_MQTT",   "Panel_Solar",
]

_DEVICE_ICONS = [
    "🌡️", "📷", "📡", "💡", "🔐", "📶", "💨", "🌿",
    "❄️", "🔌", "👁️", "🔗", "📨", "☀️",
]

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

_DEVICE_ICONS = ["📏", "🔥", "📡", "🔊", "🌡️"]


def _device_name_from_file(filename: str, index: int) -> str:
    """Return a generic IoT device name based on upload order."""
    return _DEVICE_NAMES[index % len(_DEVICE_NAMES)]


@st.cache_resource(show_spinner="Cargando modelo Conv1D…")
def _load_model():
    """Load and cache the Conv1D autoencoder from the repository."""
    import keras
    return keras.models.load_model(str(_MODEL_PATH))


@st.cache_resource(show_spinner="Cargando preprocesador…")
def _load_cached_preprocessor():
    """Return the cached IoTPreprocessor, or None if the cache file is absent."""
    if not _PREPROCESSOR_PATH.exists():
        return None
    with open(_PREPROCESSOR_PATH, "rb") as fh:
        cache = pickle.load(fh)
    return cache["preprocessor"]


def _run_inference(X_windows: np.ndarray, model) -> tuple[np.ndarray, float]:
    """Run the autoencoder and return per-window errors and threshold.

    Parameters
    ----------
    X_windows : np.ndarray
        Shape (n, window_size, features).
    model : keras.Model

    Returns
    -------
    errors : np.ndarray
        Per-window median squared error.
    threshold : float
        99th-percentile of errors.
    """
    X_pred = model.predict(X_windows, batch_size=256, verbose=0)
    errors = np.median((X_windows - X_pred) ** 2, axis=(1, 2)).astype(float)
    threshold = float(np.percentile(errors, 99))
    return errors, threshold


def _stream_inference(uploaded_file, preprocessor, model, file_size: int) -> tuple[np.ndarray, float] | None:
    """Process a large CSV in chunks to keep memory usage constant.

    Reads ``_CHUNK_SIZE`` rows at a time, transforms each chunk with the
    cached preprocessor, creates windows with a carry-over buffer so windows
    never straddle missing rows, and runs inference incrementally.  All
    per-window errors are collected in a flat list; the global threshold is
    computed once at the end from the full distribution.

    Parameters
    ----------
    uploaded_file : UploadedFile
    preprocessor : IoTPreprocessor
    model : keras.Model
    file_size : int
        Original file size in bytes, used for progress estimation.

    Returns
    -------
    (errors, threshold) or None on failure.
    """
    from windowing import create_windows

    all_errors: list[float] = []
    carry: np.ndarray | None = None
    estimated_chunks = max(file_size // (_CHUNK_SIZE * 200), 1)
    chunks_done = 0

    bar = st.progress(0, text="Procesando archivo completo…")

    try:
        for chunk in pd.read_csv(uploaded_file, chunksize=_CHUNK_SIZE, low_memory=False):
            try:
                X_chunk, _ = preprocessor.transform(chunk)
                X_arr = X_chunk.values.astype("float32")
            except Exception:
                carry = None
                continue

            if carry is not None and carry.shape[1] == X_arr.shape[1]:
                X_arr = np.vstack([carry, X_arr])

            X_windows = create_windows(X_arr, _WINDOW_SIZE)
            carry = X_arr[-(_WINDOW_SIZE - 1):]

            if X_windows.shape[0] == 0:
                continue

            X_pred = model.predict(X_windows, batch_size=512, verbose=0)
            errors = np.median((X_windows - X_pred) ** 2, axis=(1, 2)).astype(float)
            all_errors.extend(errors.tolist())

            chunks_done += 1
            progress = min(chunks_done / estimated_chunks, 0.99)
            bar.progress(progress, text=f"Ventanas procesadas: {len(all_errors):,}")

    except Exception as exc:
        bar.empty()
        st.error(f"Error durante el procesamiento: {exc}")
        return None

    bar.empty()

    if not all_errors:
        return None

    errors_arr = np.array(all_errors, dtype=float)
    threshold = float(np.percentile(errors_arr, 99))
    return errors_arr, threshold


def _preprocess_dataframes(
    dfs: list[pd.DataFrame],
) -> list[np.ndarray | None]:
    """Preprocess a list of DataFrames using the cached preprocessor or fit_transform.

    When the cache is available each DataFrame is transformed independently.
    When there is no cache, all DataFrames are concatenated, fit_transform is
    called once on the combined data, and the result is split back per device.

    Parameters
    ----------
    dfs : list of pd.DataFrame

    Returns
    -------
    list of np.ndarray or None
        One float32 array per input DataFrame; None on per-device error.
    """
    from preprocessor import IoTPreprocessor

    preprocessor = _load_cached_preprocessor()

    if preprocessor is not None:
        results: list[np.ndarray | None] = []
        for df in dfs:
            try:
                X, _ = preprocessor.transform(df)
                results.append(X.values.astype("float32"))
            except Exception:
                results.append(None)
        return results

    combined = pd.concat(dfs, ignore_index=True)
    pre = IoTPreprocessor(known_categories=None)
    X_combined, _ = pre.fit_transform(combined)
    X_arr = X_combined.values.astype("float32")

    results = []
    cursor = 0
    for df in dfs:
        n = len(df)
        results.append(X_arr[cursor: cursor + n])
        cursor += n
    return results


def _process_uploads(uploaded_files: list, model) -> dict | None:
    """Read, preprocess, and run inference on one or more uploaded files.

    Large files (> _LARGE_FILE_THRESHOLD) are processed in chunks when the
    preprocessor cache is available, keeping RAM usage constant regardless of
    file size.  Small files follow the standard in-memory path.

    The animation always shows at most _MAX_ANIMATION_WINDOWS windows (evenly
    sampled from the full results); statistics (n_windows, n_anomaly) reflect
    the complete file.

    Parameters
    ----------
    uploaded_files : list of UploadedFile
    model : keras.Model

    Returns
    -------
    dict or None
        Combined result dict on success, None on failure.
    """
    from windowing import create_windows

    preprocessor = _load_cached_preprocessor()
    devices_info: list[dict] = []
    small_dfs: list[pd.DataFrame] = []
    small_meta: list[dict] = []

    for idx, uf in enumerate(uploaded_files):
        if uf.size > _MAX_FILE_BYTES:
            st.error(f"'{uf.name}' supera el límite de 1 GB ({uf.size / 1e9:.2f} GB).")
            return None

        is_large = uf.size >= _LARGE_FILE_THRESHOLD and preprocessor is not None

        if is_large:
            st.caption(f"📦 '{uf.name}' ({uf.size / 1e6:.0f} MB) — procesando en chunks…")
            result = _stream_inference(uf, preprocessor, model, uf.size)
            if result is None:
                return None
            errors, threshold = result
            anomaly = (errors > threshold).astype(int)

            device_name = _device_name_from_file(uf.name, idx)
            devices_info.append({
                "name": device_name,
                "icon": _DEVICE_ICONS[idx % len(_DEVICE_ICONS)],
                "filename": uf.name,
                "attack_label": _ATTACK_LABELS.get(uf.name, Path(uf.name).stem.replace("_", " ")),
                "errors": errors,
                "anomaly": anomaly,
                "threshold": threshold,
                "n_windows": len(errors),
                "n_anomaly": int(anomaly.sum()),
            })
        else:
            try:
                df = pd.read_csv(uf, low_memory=False)
            except Exception as exc:
                st.error(f"No se pudo leer '{uf.name}': {exc}")
                return None
            if len(df) < _WINDOW_SIZE + 1:
                st.error(
                    f"'{uf.name}' tiene solo {len(df)} filas — "
                    f"se necesitan al menos {_WINDOW_SIZE + 1}."
                )
                return None
            small_dfs.append(df)
            small_meta.append({"name": uf.name, "index": idx})

    if small_dfs:
        try:
            arrays = _preprocess_dataframes(small_dfs)
        except Exception as exc:
            st.error(f"Error en preprocesamiento: {exc}")
            return None

        for meta, X_arr in zip(small_meta, arrays):
            if X_arr is None:
                st.warning(f"Se omitió '{meta['name']}' por error en preprocesamiento.")
                continue
            try:
                X_windows = create_windows(X_arr, _WINDOW_SIZE)
                if X_windows.shape[0] == 0:
                    st.warning(f"'{meta['name']}': no se pudieron crear ventanas.")
                    continue
                errors, threshold = _run_inference(X_windows, model)
                anomaly = (errors > threshold).astype(int)
            except Exception as exc:
                st.warning(f"'{meta['name']}': error en inferencia — {exc}")
                continue

            idx = meta["index"]
            devices_info.append({
                "name": _device_name_from_file(meta["name"], idx),
                "icon": _DEVICE_ICONS[idx % len(_DEVICE_ICONS)],
                "filename": meta["name"],
                "attack_label": _ATTACK_LABELS.get(meta["name"], Path(meta["name"]).stem.replace("_", " ")),
                "errors": errors,
                "anomaly": anomaly,
                "threshold": threshold,
                "n_windows": len(errors),
                "n_anomaly": int(anomaly.sum()),
            })

    if not devices_info:
        st.error("Ningún archivo pudo procesarse correctamente.")
        return None

    max_windows = max(d["n_windows"] for d in devices_info)
    all_windows_full: list[dict] = []
    all_errors_list: list[float] = []
    all_anomaly_list: list[int] = []
    window_devs_full: list[str] = []

    for win_idx in range(max_windows):
        for dev in devices_info:
            if win_idx < dev["n_windows"]:
                all_windows_full.append({
                    "dispositivo": dev["name"],
                    "ataque": dev["attack_label"],
                    "ventana": win_idx + 1,
                    "mse": round(float(dev["errors"][win_idx]), 8),
                    "clasificacion": "🔴 Ataque" if dev["anomaly"][win_idx] == 1 else "🟢 Normal",
                })
                all_errors_list.append(float(dev["errors"][win_idx]))
                all_anomaly_list.append(int(dev["anomaly"][win_idx]))
                window_devs_full.append(dev["name"])

    all_errors_arr = np.array(all_errors_list, dtype=float)
    all_anomaly_arr = np.array(all_anomaly_list, dtype=int)

    total = len(all_anomaly_arr)
    if total > _MAX_ANIMATION_WINDOWS:
        indices = np.linspace(0, total - 1, _MAX_ANIMATION_WINDOWS, dtype=int)
        anim_errors  = all_errors_arr[indices]
        anim_anomaly = all_anomaly_arr[indices]
        anim_devs    = [window_devs_full[i] for i in indices]
    else:
        anim_errors  = all_errors_arr
        anim_anomaly = all_anomaly_arr
        anim_devs    = window_devs_full

    return {
        "devices":     devices_info,
        "all_windows": all_windows_full,
        "all_errors":  anim_errors,
        "all_anomaly": anim_anomaly,
        "window_devs": anim_devs,
        "total_windows": total,
        "total_anomaly": int(all_anomaly_arr.sum()),
    }


def _render_device_card(col, dev: dict) -> None:
    """Render an HTML status card for one IoT device into a Streamlit column.

    Parameters
    ----------
    col : streamlit column
    dev : dict
        Device info dict from ``_process_uploads``.
    """
    n_anomaly = dev["n_anomaly"]
    n_windows = dev["n_windows"]
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
    <div style="font-size: 2rem; margin-bottom: 6px;">{dev['icon']}</div>
    <div style="color: #ecf0f1; font-weight: 600; font-size: 0.82rem;
                margin-bottom: 8px; word-break: break-word;">
        {dev['name'].replace('_', ' ')}
    </div>
    <div style="
        display: inline-block;
        background: {status_bg};
        color: white;
        font-size: 0.62rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 20px;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    ">{status_text}</div>
    <div style="color: #bdc3c7; font-size: 0.75rem; margin-bottom: 4px;">
        {dev['attack_label']}
    </div>
    <div style="color: #e74c3c; font-weight: 700; font-size: 1.1rem;">
        {n_anomaly}
        <span style="color: #7f8c8d; font-weight: 400; font-size: 0.8rem;">
            / {n_windows}
        </span>
    </div>
    <div style="color: #95a5a6; font-size: 0.72rem;">{pct:.1f}% anomalías</div>
</div>
"""
    col.markdown(html, unsafe_allow_html=True)


def _upload_prompt() -> None:
    """Render the landing screen shown when no file has been uploaded."""
    st.markdown(
        f"""
<div style="
    text-align: center;
    padding: 60px 20px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    border: 1px solid #2c3e50;
    margin: 20px 0;
">
    <div style="font-size: 4rem; margin-bottom: 16px;">🔒</div>
    <h2 style="color: #ecf0f1; margin-bottom: 8px;">Sube tus archivos de tráfico</h2>
    <p style="color: #95a5a6; font-size: 0.95rem; max-width: 520px; margin: 0 auto 16px auto;">
        Usa el panel lateral para cargar uno o varios CSV del dataset
        <b style="color:#ecf0f1;">EdgeIIoTSet</b>.
        El modelo Conv1D federado clasificará cada ventana y animará el flujo.
    </p>
    <div style="display: flex; justify-content: center; gap: 32px; margin-top: 20px; flex-wrap: wrap;">
        <div style="text-align:center;">
            <div style="font-size:1.8rem;">📄</div>
            <div style="color:#ecf0f1; font-weight:600; font-size:0.85rem;">1 archivo</div>
            <div style="color:#95a5a6; font-size:0.75rem;">Un dispositivo IoT</div>
        </div>
        <div style="color:#7f8c8d; font-size:1.5rem; align-self:center;">→</div>
        <div style="text-align:center;">
            <div style="font-size:1.8rem;">📄📄📄</div>
            <div style="color:#ecf0f1; font-weight:600; font-size:0.85rem;">hasta {_MAX_DEVICES} archivos</div>
            <div style="color:#95a5a6; font-size:0.75rem;">Varios dispositivos simultáneos</div>
        </div>
    </div>
    <div style="margin-top: 24px; color: #7f8c8d; font-size: 0.78rem;">
        Mínimo {_WINDOW_SIZE + 1} filas · Máx. 1 GB por archivo · Formato EdgeIIoTSet
    </div>
</div>
""",
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
            f"Sube **1–{_MAX_DEVICES} archivos** CSV (un dispositivo por archivo, máx. 1 GB c/u)."
        )

        uploaded_files = st.file_uploader(
            label="Selecciona CSV(s) de tráfico de red",
            type=["csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            if len(uploaded_files) > _MAX_DEVICES:
                st.warning(f"Máximo {_MAX_DEVICES} archivos a la vez — se usarán los primeros {_MAX_DEVICES}.")
                uploaded_files = uploaded_files[:_MAX_DEVICES]
            st.caption(f"{len(uploaded_files)} archivo(s) cargado(s):")
            for uf in uploaded_files:
                label = _ATTACK_LABELS.get(uf.name, uf.name)
                st.caption(f"  • {label}")

            if st.button("🔄 Limpiar y cargar otros", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key.startswith("result_"):
                        del st.session_state[key]
                st.rerun()

    st.title("🔒 IoT Network Attack Monitor")
    st.caption(
        "Detección de anomalías IoT con Conv1D autoencoder federado — "
        "EdgeIIoTSet dataset · 14 tipos de ataque"
    )

    if not uploaded_files:
        _upload_prompt()
        return

    cache_key = "result_" + "_".join(f"{uf.name}_{uf.size}" for uf in uploaded_files)
    if cache_key not in st.session_state:
        with st.spinner(f"⚙️ Procesando {len(uploaded_files)} archivo(s)…"):
            st.session_state[cache_key] = _process_uploads(uploaded_files, model)

    data = st.session_state[cache_key]

    if data is None:
        return

    devices_info   = data["devices"]
    all_errors     = data["all_errors"]
    all_anomaly    = data["all_anomaly"]
    window_devs    = data["window_devs"]
    all_windows    = data["all_windows"]

    total_attacks       = data["total_anomaly"]
    total_windows       = data["total_windows"]
    devs_under_attack   = sum(1 for d in devices_info if d["n_anomaly"] > 0)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Dispositivos",              len(devices_info))
    col_m2.metric("Ventanas analizadas",       f"{total_windows:,}")
    col_m3.metric("Ataques detectados",        f"{total_attacks:,}")
    col_m4.metric("Dispositivos bajo ataque",  devs_under_attack)

    st.markdown("---")

    from attack_animation import build_attack_animation

    device_attack_types = {d["name"]: d["attack_label"] for d in devices_info}

    st.plotly_chart(
        build_attack_animation(
            device_names=[d["name"] for d in devices_info],
            anomaly=all_anomaly,
            errors=all_errors,
            window_devs=window_devs,
            device_attack_types=device_attack_types,
        ),
        use_container_width=True,
    )

    st.markdown("#### Estado de Dispositivos")
    n_cols = min(len(devices_info), _MAX_DEVICES)
    card_cols = st.columns(n_cols)
    for idx, dev in enumerate(devices_info):
        _render_device_card(card_cols[idx % n_cols], dev)

    st.markdown("---")
    st.markdown("#### Registro de Clasificación")
    st.dataframe(
        pd.DataFrame(all_windows),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
