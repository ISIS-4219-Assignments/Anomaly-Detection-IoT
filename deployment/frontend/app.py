"""app.py

Live network-attack simulation dashboard for the FL IoT Anomaly Detection system.

Upload one or more CSV files in EdgeIIoTSet format (preferably test/mixed-traffic files
so the classifier can show both normal and attack classifications):
- 1 file  → single device, treated as one traffic stream.
- 2–14 files → one device per file; the animation shows all simultaneously.

Inference pipeline
------------------
1. Upload CSV(s)  →  IoTPreprocessor.transform() (or fit_transform if no cache)
2. create_windows(window_size=30) per device
3. model.predict()  →  per-window median squared error
4. threshold = 99th-percentile of reconstruction errors (computed per device)
5. anomaly = mse > threshold
6. Live streaming replay of windows with real-time classification badges
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SIM_DIR = _PROJECT_ROOT / "src" / "simulation"
_MODEL_PATH = _PROJECT_ROOT / "src" / "models" / "conv1d_autoencoder_final.keras"
_PREPROCESSOR_PATH = Path(__file__).resolve().parent.parent / "preprocessor_cache.pkl"
_THRESHOLD_PATH = Path(__file__).resolve().parent.parent / "threshold_cache.pkl"
_WINDOW_SIZE = 30
_MAX_DEVICES = 14
_MAX_FILE_BYTES = 1 * 1024 * 1024 * 1024
_MAX_ANIMATION_WINDOWS = 300
_CHUNK_SIZE = 5_000
_LARGE_FILE_THRESHOLD = 50 * 1024 * 1024  # 50 MB
_CHART_UPDATE_INTERVAL = 0.12  # seconds between chart refreshes (~8 fps)

sys.path.insert(0, str(_SIM_DIR))

_DEVICE_NAMES = [
    "Termostato",    "Camara_IP",    "Sensor_Temp",  "Medidor_Luz",
    "Cerradura",     "Router_IoT",   "Sensor_Gas",   "Monitor_CO2",
    "Control_HVAC",  "Interruptor",  "Sensor_Mov",   "Gateway",
    "Broker_MQTT",   "Panel_Solar",
]

_DEVICE_ICONS = [
    "[THR]", "[CAM]", "[SNS]", "[LUZ]", "[CER]", "[RTR]", "[GAS]", "[CO2]",
    "[HVC]", "[INT]", "[MOV]", "[GWY]", "[MQT]", "[SOL]",
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

_SPLIT_DEVICE_NAMES = {
    "distance":                  "Sensor_Distancia",
    "flame_sensor":              "Sensor_Llama",
    "ir_receiver":               "Receptor_IR",
    "phvalue":                   "Sensor_pH",
    "soil_moisture":             "Sensor_Humedad",
    "sound_sensor":              "Sensor_Sonido",
    "temperature_and_humidity":  "Sensor_TempHum",
    "water_level":               "Sensor_Agua",
}


def _device_name_from_file(filename: str, index: int) -> str:
    """Return an IoT device name: folder-based for known split files, generic otherwise."""
    stem = Path(filename).stem.lower().replace(" ", "_")
    if stem in _SPLIT_DEVICE_NAMES:
        return _SPLIT_DEVICE_NAMES[stem]
    for key, name in _SPLIT_DEVICE_NAMES.items():
        if key in stem:
            return name
    return _DEVICE_NAMES[index % len(_DEVICE_NAMES)]


@st.cache_resource(show_spinner="Cargando modelo Conv1D...")
def _load_model():
    """Load and cache the Conv1D autoencoder from the repository."""
    import keras
    return keras.models.load_model(str(_MODEL_PATH))


@st.cache_resource(show_spinner="Cargando preprocesador...")
def _load_cached_preprocessor():
    """Return the cached IoTPreprocessor, or None if the cache file is absent."""
    if not _PREPROCESSOR_PATH.exists():
        return None
    with open(_PREPROCESSOR_PATH, "rb") as fh:
        cache = pickle.load(fh)
    return cache["preprocessor"]


@st.cache_resource
def _load_threshold_cache() -> dict | None:
    """Return the threshold cache dict, or None if the file is absent.

    The cache contains:
    - ``threshold``: global value (99th pct pooled across all 8 devices).
    - ``per_device``: per-device values keyed by device folder name.

    Both were read directly from results/conv1d_*_report.txt — the same
    values used during FL evaluation, so they are genuine model properties.
    """
    if not _THRESHOLD_PATH.exists():
        return None
    with open(_THRESHOLD_PATH, "rb") as fh:
        return pickle.load(fh)


def _resolve_threshold(cache: dict | None, filename: str) -> float | None:
    """Return the best threshold for a given uploaded filename.

    Prefers a per-device threshold when the filename matches a known device
    folder name (e.g. 'distance_test.csv' → Distance → 0.0007), then falls
    back to the global threshold, then None.
    """
    if cache is None:
        return None
    per_device: dict = cache.get("per_device", {})
    stem = Path(filename).stem.lower().replace(" ", "_")
    for device_key in per_device:
        if device_key.lower() in stem:
            return float(per_device[device_key])
    return float(cache["threshold"])


def _run_inference(
    X_windows: np.ndarray,
    model,
    fixed_threshold: float | None = None,
) -> tuple[np.ndarray, float]:
    """Run the autoencoder and return per-window reconstruction errors and threshold.

    Priority order for the threshold:
    1. ``fixed_threshold`` — pre-computed from normal val data (best, model-derived).
    2. Fallback: 99th percentile of the uploaded data's own errors (always flags
       exactly 1%, regardless of whether the data is attack or normal).
    """
    X_pred = model.predict(X_windows, batch_size=256, verbose=0)
    errors = np.median((X_windows - X_pred) ** 2, axis=(1, 2)).astype(float)
    threshold = fixed_threshold if fixed_threshold is not None else float(np.percentile(errors, 99))
    return errors, threshold


def _stream_inference(
    uploaded_file,
    preprocessor,
    model,
    file_size: int,
    fixed_threshold: float | None = None,
) -> tuple[np.ndarray, float] | None:
    """Process a large CSV in chunks to keep memory usage constant."""
    from windowing import create_windows

    all_errors: list[float] = []
    carry: np.ndarray | None = None
    estimated_chunks = max(file_size // (_CHUNK_SIZE * 200), 1)
    chunks_done = 0

    bar = st.progress(0, text="Procesando archivo completo...")

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
    threshold = fixed_threshold if fixed_threshold is not None else float(np.percentile(errors_arr, 99))
    return errors_arr, threshold


def _preprocess_dataframes(dfs: list[pd.DataFrame]) -> list[np.ndarray | None]:
    """Preprocess a list of DataFrames using the cached preprocessor or fit_transform."""
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


def _process_uploads(uploaded_files: list, model, threshold_cache: dict | None) -> dict | None:
    """Read, preprocess, and run inference on one or more uploaded files.

    ``threshold_cache`` comes from threshold_cache.pkl and holds per-device and
    global threshold values derived from the FL training results.  Each file gets
    the most specific matching threshold; unknown files use the global value.
    If None, falls back to the 99th percentile of each device's own errors.
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

        dev_threshold = _resolve_threshold(threshold_cache, uf.name)
        is_large = uf.size >= _LARGE_FILE_THRESHOLD and preprocessor is not None

        if is_large:
            st.caption(f"'{uf.name}' ({uf.size / 1e6:.0f} MB) — procesando en chunks...")
            result = _stream_inference(uf, preprocessor, model, uf.size, dev_threshold)
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
            small_meta.append({"name": uf.name, "index": idx, "threshold": dev_threshold})

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
                errors, threshold = _run_inference(X_windows, model, meta["threshold"])
                anomaly = (errors > threshold).astype(int)
            except Exception as exc:
                st.warning(f"'{meta['name']}': error en inferencia — {exc}")
                continue

            idx = meta["index"]
            attack_label = _ATTACK_LABELS.get(meta["name"], Path(meta["name"]).stem.replace("_", " "))
            devices_info.append({
                "name": _device_name_from_file(meta["name"], idx),
                "icon": _DEVICE_ICONS[idx % len(_DEVICE_ICONS)],
                "filename": meta["name"],
                "attack_label": attack_label,
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
                    "clasificacion": "Ataque" if dev["anomaly"][win_idx] == 1 else "Normal",
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
        anim_windows = [all_windows_full[i] for i in indices]
    else:
        anim_errors  = all_errors_arr
        anim_anomaly = all_anomaly_arr
        anim_devs    = window_devs_full
        anim_windows = all_windows_full

    return {
        "devices":       devices_info,
        "all_windows":   all_windows_full,
        "anim_errors":   anim_errors,
        "anim_anomaly":  anim_anomaly,
        "anim_devs":     anim_devs,
        "anim_windows":  anim_windows,
        "total_windows": total,
        "total_anomaly": int(all_anomaly_arr.sum()),
    }


def _live_mse_chart(errors: list[float], anomaly: list[int]) -> go.Figure:
    """Build a real-time MSE line chart with red/green markers per classification."""
    colors = ["#e74c3c" if a else "#2ecc71" for a in anomaly]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=errors,
        mode="lines",
        line=dict(color="rgba(52,152,219,0.35)", width=1.2),
        showlegend=False,
        hoverinfo="none",
    ))
    fig.add_trace(go.Scatter(
        y=errors,
        mode="markers",
        marker=dict(color=colors, size=5, opacity=0.85, line=dict(width=0)),
        showlegend=False,
        hovertemplate="MSE: %{y:.6f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=200,
        margin=dict(l=50, r=10, t=10, b=30),
        xaxis=dict(title="Ventana #", showgrid=True, gridcolor="#1e2d3d", color="#7f8c8d"),
        yaxis=dict(title="MSE", type="log", showgrid=True, gridcolor="#1e2d3d", color="#7f8c8d"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _live_cards_html(devices_info: list[dict], live_counts: dict) -> str:
    """Build device status cards HTML reflecting live counters during streaming."""
    cards = []
    for dev in devices_info:
        name = dev["name"]
        lc = live_counts[name]
        n_win = lc["n_windows"]
        n_atk = lc["n_anomaly"]
        pct = 100.0 * n_atk / n_win if n_win > 0 else 0.0
        border = "#e74c3c" if n_atk > 0 else "#2ecc71"
        status = "ATAQUE" if n_atk > 0 else "OK"
        status_color = "#e74c3c" if n_atk > 0 else "#2ecc71"
        cards.append(f"""
<div style="
    background:linear-gradient(135deg,#1a1a2e,#16213e);
    border:2px solid {border};
    border-radius:10px;
    padding:10px 8px;
    min-width:90px;
    text-align:center;
    flex:1;
">
    <div style="font-size:0.7rem;font-weight:700;color:#95a5a6;">{dev['icon']}</div>
    <div style="color:#ecf0f1;font-size:0.68rem;font-weight:600;margin:2px 0;">
        {name.replace('_', ' ')}
    </div>
    <div style="color:{status_color};font-size:0.6rem;font-weight:700;margin-bottom:4px;">
        {status}
    </div>
    <div style="color:#e74c3c;font-size:0.9rem;font-weight:700;">
        {n_atk}<span style="color:#7f8c8d;font-size:0.65rem;">/{n_win}</span>
    </div>
    <div style="color:#95a5a6;font-size:0.62rem;">{pct:.0f}% anomalías</div>
</div>""")

    inner = "\n".join(cards)
    return f'<div style="display:flex;gap:6px;flex-wrap:wrap;align-items:stretch;">{inner}</div>'


def _run_live_simulation(data: dict, speed_wps: int) -> None:
    """Replay classification results window-by-window with real-time UI updates."""
    anim_errors  = data["anim_errors"]
    anim_anomaly = data["anim_anomaly"]
    anim_devs    = data["anim_devs"]
    anim_windows = data["anim_windows"]
    devices_info = data["devices"]

    live_counts = {d["name"]: {"n_windows": 0, "n_anomaly": 0} for d in devices_info}
    delay = 1.0 / max(speed_wps, 1)
    n_total = len(anim_errors)

    badge_ph = st.empty()

    col_chart, col_now = st.columns([3, 1])
    with col_chart:
        st.caption("Error de reconstruccion (MSE) — Ataque (rojo)  Normal (verde)")
        chart_ph = st.empty()
    with col_now:
        st.caption("Última clasificación")
        now_ph = st.empty()

    st.caption("Estado de dispositivos")
    cards_ph = st.empty()

    st.caption("Registro de clasificaciones (últimas 25)")
    log_ph = st.empty()

    shown_errors: list[float] = []
    shown_anomaly: list[int] = []
    shown_log: list[dict] = []
    last_ui_update = 0.0

    for i, (err, anom, dev, win) in enumerate(
        zip(anim_errors, anim_anomaly, anim_devs, anim_windows)
    ):
        shown_errors.append(float(err))
        shown_anomaly.append(int(anom))
        shown_log.append(win)

        live_counts[dev]["n_windows"] += 1
        if anom:
            live_counts[dev]["n_anomaly"] += 1

        badge_color = "#e74c3c" if anom else "#2ecc71"
        badge_label = "ATAQUE DETECTADO" if anom else "Normal"
        progress_pct = int(100 * (i + 1) / n_total)
        badge_ph.markdown(
            f"""
<div style="
    background:{badge_color}18;
    border:2px solid {badge_color};
    border-radius:10px;
    padding:10px 20px;
    margin-bottom:6px;
    display:flex;
    align-items:center;
    gap:16px;
">
    <div>
        <div style="color:{badge_color};font-weight:800;font-size:1.15rem;">{badge_label}</div>
        <div style="color:#bdc3c7;font-size:0.82rem;">
            Ventana <b>#{i+1}</b> de {n_total} &nbsp;·&nbsp;
            Dispositivo: <b>{dev.replace('_',' ')}</b> &nbsp;·&nbsp;
            MSE: <code>{err:.6f}</code>
        </div>
    </div>
    <div style="margin-left:auto;color:#7f8c8d;font-size:0.78rem;">{progress_pct}% completado</div>
</div>""",
            unsafe_allow_html=True,
        )

        now = time.time()
        if now - last_ui_update >= _CHART_UPDATE_INTERVAL or i == n_total - 1:
            chart_ph.plotly_chart(
                _live_mse_chart(shown_errors, shown_anomaly),
                use_container_width=True,
            )

            now_color = "#e74c3c" if anom else "#2ecc71"
            now_ph.markdown(
                f"""
<div style="
    background:{now_color}22;
    border:1px solid {now_color};
    border-radius:8px;
    padding:12px 8px;
    text-align:center;
">
    <div style="font-size:1.8rem;">{"[!]" if anom else "[OK]"}</div>
    <div style="color:{now_color};font-weight:700;font-size:0.9rem;">
        {"Ataque" if anom else "Normal"}
    </div>
    <div style="color:#bdc3c7;font-size:0.72rem;">{dev.replace('_',' ')}</div>
    <div style="color:#95a5a6;font-size:0.68rem;">MSE {err:.3e}</div>
</div>""",
                unsafe_allow_html=True,
            )

            cards_ph.markdown(_live_cards_html(devices_info, live_counts), unsafe_allow_html=True)
            log_ph.dataframe(
                pd.DataFrame(shown_log[-25:]),
                use_container_width=True,
                hide_index=True,
            )
            last_ui_update = now

        time.sleep(delay)

    total_attacks = sum(lc["n_anomaly"] for lc in live_counts.values())
    badge_ph.markdown(
        f"""
<div style="
    background:#27ae6022;
    border:2px solid #27ae60;
    border-radius:10px;
    padding:10px 20px;
    margin-bottom:6px;
">
    <span style="color:#27ae60;font-weight:800;font-size:1.05rem;">Simulacion completada</span>
    <span style="color:#bdc3c7;margin-left:12px;font-size:0.85rem;">
        {n_total} ventanas analizadas · <b>{total_attacks}</b> ataques detectados
    </span>
</div>""",
        unsafe_allow_html=True,
    )


def _render_device_card(col, dev: dict) -> None:
    """Render an HTML status card for one IoT device into a Streamlit column."""
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
    <div style="font-size: 0.85rem; font-weight: 700; margin-bottom: 6px; color: #95a5a6;">{dev['icon']}</div>
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
    <h2 style="color: #ecf0f1; margin-bottom: 8px;">Sube tus archivos de trafico</h2>
    <p style="color: #95a5a6; font-size: 0.95rem; max-width: 560px; margin: 0 auto 16px auto;">
        Usa el panel lateral para cargar uno o varios CSV del dataset
        <b style="color:#ecf0f1;">EdgeIIoTSet</b>.
        Para ver clasificaciones mixtas (normal + ataque), sube archivos de
        <b style="color:#3498db;">test o combinados</b>.
        Con archivos de ataque puro, todas las ventanas resultaran anomalias.
    </p>
    <div style="display: flex; justify-content: center; gap: 32px; margin-top: 20px; flex-wrap: wrap;">
        <div style="text-align:center;">
            <div style="color:#3498db; font-weight:600; font-size:0.85rem;">Recomendado</div>
            <div style="color:#95a5a6; font-size:0.75rem;">CSV de test / trafico mixto</div>
        </div>
        <div style="color:#7f8c8d; font-size:1.5rem; align-self:center;">·</div>
        <div style="text-align:center;">
            <div style="color:#ecf0f1; font-weight:600; font-size:0.85rem;">Tambien valido</div>
            <div style="color:#95a5a6; font-size:0.75rem;">CSV de ataque puro (100% anomalias)</div>
        </div>
        <div style="color:#7f8c8d; font-size:1.5rem; align-self:center;">·</div>
        <div style="text-align:center;">
            <div style="color:#ecf0f1; font-weight:600; font-size:0.85rem;">hasta {_MAX_DEVICES} archivos</div>
            <div style="color:#95a5a6; font-size:0.75rem;">Varios dispositivos simultaneos</div>
        </div>
    </div>
    <div style="margin-top: 24px; color: #7f8c8d; font-size: 0.78rem;">
        Mínimo {_WINDOW_SIZE + 1} filas · Máx. 1 GB por archivo · Formato EdgeIIoTSet
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_final_network(data: dict) -> None:
    """Render the full Plotly network animation after simulation completes."""
    from attack_animation import build_attack_animation

    devices_info = data["devices"]
    device_attack_types = {d["name"]: d["attack_label"] for d in devices_info}

    st.plotly_chart(
        build_attack_animation(
            device_names=[d["name"] for d in devices_info],
            anomaly=data["anim_anomaly"],
            errors=data["anim_errors"],
            window_devs=data["anim_devs"],
            device_attack_types=device_attack_types,
        ),
        use_container_width=True,
    )


def main() -> None:
    """Entry point for the IoT network-attack simulation Streamlit dashboard."""
    st.set_page_config(
        page_title="IoT Attack Monitor",
        page_icon=":lock:",
        layout="wide",
    )

    model = _load_model()

    with st.sidebar:
        st.markdown("## IoT Attack Monitor")
        st.divider()

        st.markdown("### Modelo")
        st.success("Conv1D Autoencoder — FL")
        st.metric("AUROC",    "0.9999")
        st.metric("Recall",   "99.95 %")
        st.metric("F1 (bal)", "0.9986")
        st.caption("Entrenado con Federated Learning sobre 8 dispositivos IoT.")

        st.divider()

        st.markdown("### Velocidad de simulacion")
        speed_wps = st.select_slider(
            "Ventanas por segundo",
            options=[2, 5, 10, 20, 50, 100],
            value=10,
            label_visibility="collapsed",
        )
        st.caption(f"**{speed_wps} ventanas/seg** — la simulacion muestra cada clasificacion en tiempo real.")

        st.divider()

        st.markdown("### Cargar datos")
        st.caption(
            f"Sube **1-{_MAX_DEVICES} archivos** CSV (un dispositivo por archivo, max. 1 GB c/u). "
            "Para ver trafico mixto, usa archivos de **test** o combinados del dataset."
        )

        uploaded_files = st.file_uploader(
            label="Selecciona CSV(s) de trafico de red",
            type=["csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            if len(uploaded_files) > _MAX_DEVICES:
                st.warning(f"Maximo {_MAX_DEVICES} archivos a la vez — se usaran los primeros {_MAX_DEVICES}.")
                uploaded_files = uploaded_files[:_MAX_DEVICES]
            st.caption(f"{len(uploaded_files)} archivo(s) cargado(s):")
            for uf in uploaded_files:
                label = _ATTACK_LABELS.get(uf.name, uf.name)
                st.caption(f"  - {label}")

            if st.button("Limpiar y cargar otros", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key.startswith("result_") or key.endswith("_sim_done") or key.endswith("_sim_started"):
                        del st.session_state[key]
                st.rerun()

    st.title("IoT Network Attack Monitor")
    st.caption(
        "Deteccion de anomalias IoT con Conv1D autoencoder federado — "
        "EdgeIIoTSet dataset · 14 tipos de ataque"
    )

    if not uploaded_files:
        _upload_prompt()
        return

    threshold_cache = _load_threshold_cache()

    cache_key = "result_" + "_".join(f"{uf.name}_{uf.size}" for uf in uploaded_files)
    if cache_key not in st.session_state:
        with st.spinner(f"Procesando {len(uploaded_files)} archivo(s)..."):
            st.session_state[cache_key] = _process_uploads(uploaded_files, model, threshold_cache)

    data = st.session_state[cache_key]
    if data is None:
        return

    devices_info = data["devices"]
    total_attacks = data["total_anomaly"]
    total_windows = data["total_windows"]
    devs_under_attack = sum(1 for d in devices_info if d["n_anomaly"] > 0)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Dispositivos",             len(devices_info))
    col_m2.metric("Ventanas analizadas",      f"{total_windows:,}")
    col_m3.metric("Ataques detectados",       f"{total_attacks:,}")
    col_m4.metric("Dispositivos bajo ataque", devs_under_attack)

    if threshold_cache is not None:
        global_thr = threshold_cache.get("threshold", "-")
        st.success(
            f"**Umbral del modelo cargado** (global: `{global_thr}`, "
            "por dispositivo: 0.0006-0.0008) — leido directamente de los resultados de entrenamiento FL. "
            "Clasificaciones absolutas, no relativas a los datos subidos."
        )
    else:
        st.warning(
            "**Umbral relativo** — `threshold_cache.pkl` no encontrado. "
            "Siempre se marca el 1% con mayor MSE independientemente del contenido."
        )

    st.markdown("---")

    sim_done_key = cache_key + "_sim_done"
    sim_started_key = cache_key + "_sim_started"

    if sim_done_key not in st.session_state and sim_started_key not in st.session_state:
        st.info("Archivos procesados. Presiona 'Iniciar simulacion' para comenzar.")
        if st.button("Iniciar simulacion", type="primary", use_container_width=False):
            st.session_state[sim_started_key] = True
            st.rerun()

    elif sim_started_key in st.session_state and sim_done_key not in st.session_state:
        st.markdown("### Clasificando trafico en tiempo real...")
        _run_live_simulation(data, speed_wps)
        st.session_state[sim_done_key] = True
        del st.session_state[sim_started_key]
        st.rerun()

    else:
        col_btn, _ = st.columns([1, 4])
        if col_btn.button("Repetir simulacion", use_container_width=True):
            del st.session_state[sim_done_key]
            st.rerun()

        st.markdown("#### Topologia de red — reproduccion completa")
        _render_final_network(data)

        st.markdown("#### Estado final de dispositivos")
        n_cols = min(len(devices_info), _MAX_DEVICES)
        card_cols = st.columns(n_cols)
        for idx, dev in enumerate(devices_info):
            _render_device_card(card_cols[idx % n_cols], dev)

        st.markdown("---")
        st.markdown("#### Registro completo de clasificaciones")
        st.dataframe(
            pd.DataFrame(data["all_windows"]),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
