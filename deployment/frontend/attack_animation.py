"""attack_animation.py

Plotly-based animated network visualization for the FL IoT Anomaly Detection
system.

Topology: IoT device nodes sit on a circle around a central Server node.
Each device receives traffic (normal or attack) and forwards it to the server.
Packets travel FROM each device TOWARD the central server.

  Red    — anomaly window (reconstruction error > threshold)
  Green  — normal window

Device nodes accumulate colour from green to red as attack packets are sent,
showing which devices are forwarding the most malicious traffic.  The server
node pulses red when it is actively receiving an attack packet.
"""

import math
from typing import Sequence

import numpy as np
import plotly.graph_objects as go

_RADIUS = 2.2
_PIPELINE = 7
_FRAME_MS = 110
_COLOR_NORMAL = "#27ae60"
_COLOR_ANOMALY = "#e74c3c"
_COLOR_IDLE = "#2c3e50"
_COLOR_DEVICE_SAFE = "#2ecc71"
_COLOR_DEVICE_HOT = "#e74c3c"
_COLOR_SERVER_IDLE = "#2980b9"
_COLOR_SERVER_HOT = "#e74c3c"


def _circle(n: int) -> list[tuple[float, float]]:
    """Return n evenly-spaced (x, y) positions on a circle, starting at top."""
    return [
        (
            _RADIUS * math.cos(2 * math.pi * i / n - math.pi / 2),
            _RADIUS * math.sin(2 * math.pi * i / n - math.pi / 2),
        )
        for i in range(n)
    ]


def _lerp_hex(t: float, c0: str, c1: str) -> str:
    """Interpolate between two hex colours; t in [0, 1]."""
    def decode(c: str) -> tuple[int, int, int]:
        h = c.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    r0, g0, b0 = decode(c0)
    r1, g1, b1 = decode(c1)
    return "#{:02x}{:02x}{:02x}".format(
        int(r0 + (r1 - r0) * t),
        int(g0 + (g1 - g0) * t),
        int(b0 + (b1 - b0) * t),
    )


def _edge_trace(dev_pos: dict[str, tuple[float, float]]) -> go.Scatter:
    """Single static trace for all device-to-server edges."""
    x: list = []
    y: list = []
    for dx, dy in dev_pos.values():
        x += [dx, 0.0, None]
        y += [dy, 0.0, None]
    return go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color="#dfe6e9", width=1.2),
        hoverinfo="none",
        showlegend=False,
    )


def _label_trace(
    dev_pos: dict[str, tuple[float, float]],
    device_attack_types: dict | None = None,
) -> go.Scatter:
    """Static device-name labels positioned below each node."""
    if device_attack_types:
        labels = [
            f"{n.replace('_', ' ')}<br><span style='font-size:9px'>{device_attack_types.get(n, '')}</span>"
            for n in dev_pos
        ]
    else:
        labels = [n.replace("_", " ") for n in dev_pos]
    xs = [pos[0] for pos in dev_pos.values()]
    ys = [pos[1] - 0.38 for pos in dev_pos.values()]
    return go.Scatter(
        x=xs, y=ys,
        mode="text",
        text=labels,
        textfont=dict(size=10, color="#2c3e50"),
        hoverinfo="none",
        showlegend=False,
    )


def _packet_trace(packets: list[dict]) -> go.Scatter:
    """Dynamic trace for in-flight packets."""
    if not packets:
        return go.Scatter(x=[], y=[], mode="markers", showlegend=False, hoverinfo="none",
                          marker=dict(size=[], color=[]))
    return go.Scatter(
        x=[p["x"] for p in packets],
        y=[p["y"] for p in packets],
        mode="markers",
        marker=dict(
            color=[p["color"] for p in packets],
            size=[p["size"] for p in packets],
            opacity=0.88,
            line=dict(width=1.5, color="white"),
        ),
        hoverinfo="none",
        showlegend=False,
    )


def _device_trace(
    dev_pos: dict[str, tuple[float, float]],
    attack_counts: dict[str, int],
    max_attacks: int,
) -> go.Scatter:
    """Dynamic trace for device nodes; colour reflects accumulated attack count."""
    colors = [
        _lerp_hex(
            min(attack_counts[d] / max(max_attacks, 1), 1.0),
            _COLOR_DEVICE_SAFE,
            _COLOR_DEVICE_HOT,
        )
        for d in dev_pos
    ]
    hover = [
        f"<b>{d.replace('_',' ')}</b><br>Paquetes de ataque enviados: {attack_counts[d]}"
        for d in dev_pos
    ]
    return go.Scatter(
        x=[pos[0] for pos in dev_pos.values()],
        y=[pos[1] for pos in dev_pos.values()],
        mode="markers",
        marker=dict(size=40, color=colors, line=dict(width=2.5, color="white")),
        hovertext=hover,
        hoverinfo="text",
        showlegend=False,
    )


def _server_trace(receiving_attack: bool) -> go.Scatter:
    """Dynamic trace for the central server node."""
    color = _COLOR_SERVER_HOT if receiving_attack else _COLOR_SERVER_IDLE
    return go.Scatter(
        x=[0], y=[0],
        mode="markers+text",
        marker=dict(size=52, color=color, symbol="square",
                    line=dict(width=3, color="white")),
        text=["🗄️"],
        textposition="middle center",
        textfont=dict(size=20),
        hovertext=["<b>Servidor Central</b>"],
        hoverinfo="text",
        showlegend=False,
    )


def build_attack_animation(
    device_names: Sequence[str],
    anomaly: np.ndarray,
    errors: np.ndarray,
    window_devs: Sequence[str] | None = None,
    device_attack_types: dict[str, str] | None = None,
) -> go.Figure:
    """Build the animated IoT network Plotly figure.

    Packets flow FROM each device TOWARD the central server.  Red packets
    carry anomalous traffic; green packets carry normal traffic.

    Parameters
    ----------
    device_names : sequence of str
        IoT device names to display on the circle.
    anomaly : np.ndarray
        Binary anomaly labels per inference window (0 = normal, 1 = attack).
    errors : np.ndarray
        Reconstruction MSE per window; used to scale packet size.
    window_devs : sequence of str, optional
        Per-window device assignment.  When provided each packet is sent by
        the correct device; otherwise round-robin is used.
    device_attack_types : dict, optional
        Mapping device name → attack-type label shown under the node.

    Returns
    -------
    go.Figure
        Self-contained Plotly figure with Play / Pause controls.
    """
    device_names = list(device_names)
    dev_pos = dict(zip(device_names, _circle(len(device_names))))

    n_win = min(int(len(anomaly)), 100)
    anomaly = np.asarray(anomaly[:n_win], dtype=int)
    errors = np.asarray(errors[:n_win], dtype=float)
    max_err = float(np.max(errors)) if errors.size > 0 else 1.0

    if window_devs is not None:
        window_dev = [str(d) for d in window_devs[:n_win]]
    else:
        window_dev = [device_names[i % len(device_names)] for i in range(n_win)]

    arrival: dict[int, int] = {i: i + _PIPELINE for i in range(n_win)}
    total_frames = n_win + _PIPELINE

    static_edges = _edge_trace(dev_pos)
    static_labels = _label_trace(dev_pos, device_attack_types=device_attack_types)

    def _frame_content(f: int, counts: dict[str, int]) -> tuple:
        """Compute packet, device, and server traces for a single animation frame."""
        packets: list[dict] = []
        server_receiving_attack = False

        for i in range(n_win):
            if i <= f < i + _PIPELINE:
                progress = (f - i) / _PIPELINE
                source = window_dev[i]
                if source not in dev_pos:
                    continue
                dx, dy = dev_pos[source]
                px = dx * (1 - progress)
                py = dy * (1 - progress)
                is_anom = bool(anomaly[i])
                norm = float(min(errors[i] / max_err, 1.0))
                packets.append({
                    "x": px, "y": py,
                    "color": _COLOR_ANOMALY if is_anom else _COLOR_NORMAL,
                    "size": int(9 + norm * 10),
                })
                if is_anom and progress > 0.75:
                    server_receiving_attack = True

        max_attacks = max(counts.values()) if counts else 1

        return (
            _packet_trace(packets),
            _device_trace(dev_pos, counts, max_attacks),
            _server_trace(server_receiving_attack),
        )

    p0, d0, s0 = _frame_content(0, {d: 0 for d in device_names})
    base_data = [static_edges, static_labels, p0, d0, s0]

    frames: list[go.Frame] = []
    running: dict[str, int] = {d: 0 for d in device_names}

    for f in range(total_frames):
        for i in range(n_win):
            if arrival[i] == f and anomaly[i] == 1:
                running[window_dev[i]] += 1

        pt, dt, st = _frame_content(f, dict(running))
        frames.append(go.Frame(
            data=[static_edges, static_labels, pt, dt, st],
            name=str(f),
        ))

    fig = go.Figure(data=base_data, frames=frames)

    slider_steps = [
        {
            "args": [[str(f)], {"frame": {"duration": _FRAME_MS, "redraw": True},
                                "mode": "immediate", "transition": {"duration": 0}}],
            "label": "",
            "method": "animate",
        }
        for f in range(total_frames)
    ]

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.6, 3.6]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.6, 3.6],
                   scaleanchor="x"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=520,
        margin=dict(l=10, r=10, t=50, b=60),
        title=dict(
            text="Red IoT → Servidor Central — 🔴 Ataque  🟢 Normal",
            font=dict(size=14, color="#2c3e50"),
            x=0.5,
        ),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "y": -0.08,
            "x": 0.5,
            "xanchor": "center",
            "buttons": [
                {
                    "label": "▶  Reproducir",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": _FRAME_MS, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0}}],
                },
                {
                    "label": "⏸  Pausar",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                },
            ],
        }],
        sliders=[{
            "steps": slider_steps,
            "x": 0.05,
            "len": 0.9,
            "y": -0.04,
            "currentvalue": {"visible": False},
            "transition": {"duration": 0},
        }],
    )

    return fig
