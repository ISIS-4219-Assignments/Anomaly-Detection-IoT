"""network_graph.py

Plotly-based federated learning network topology visualization.

Renders a server node at the center surrounded by device nodes on a circle.
Node colors reflect the current training state of each participant.
Animated packet markers travel along edges to visualize weight uploads.
"""

import math
import plotly.graph_objects as go

_RADIUS = 2.3
_SERVER_X, _SERVER_Y = 0.0, 0.0

_STATE_COLORS: dict[str, str] = {
    "idle": "#95a5a6",
    "ready": "#bdc3c7",
    "training": "#e67e22",
    "uploading": "#2980b9",
    "aggregating": "#8e44ad",
    "done": "#27ae60",
}

_STATE_LABELS: dict[str, str] = {
    "idle": "En espera",
    "ready": "Listo",
    "training": "Entrenando",
    "uploading": "Enviando pesos",
    "aggregating": "Agregando",
    "done": "Completado",
}


def _circle_positions(n: int) -> list[tuple[float, float]]:
    """Return n evenly-spaced (x, y) positions on a circle, starting at the top."""
    return [
        (
            _RADIUS * math.cos(2 * math.pi * i / n - math.pi / 2),
            _RADIUS * math.sin(2 * math.pi * i / n - math.pi / 2),
        )
        for i in range(n)
    ]


def _edge_traces(
    device_pos: dict[str, tuple[float, float]],
    device_states: dict[str, str],
) -> list[go.Scatter]:
    """Build one Scatter trace per device edge, styled by upload activity."""
    traces = []
    for name, (dx, dy) in device_pos.items():
        state = device_states.get(name, "idle")
        active = state in ("uploading", "aggregating")
        traces.append(go.Scatter(
            x=[dx, _SERVER_X, None],
            y=[dy, _SERVER_Y, None],
            mode="lines",
            line=dict(
                color="#2980b9" if active else "#dfe6e9",
                width=2.5 if active else 1.2,
                dash="dot" if active else "solid",
            ),
            hoverinfo="none",
            showlegend=False,
        ))
    return traces


def _packet_traces(
    device_pos: dict[str, tuple[float, float]],
    packet_progress: dict[str, float],
) -> list[go.Scatter]:
    """Build one Scatter trace per in-flight data packet."""
    traces = []
    for name, progress in packet_progress.items():
        if 0.0 < progress <= 1.05 and name in device_pos:
            dx, dy = device_pos[name]
            t = min(progress, 1.0)
            px = dx + (_SERVER_X - dx) * t
            py = dy + (_SERVER_Y - dy) * t
            traces.append(go.Scatter(
                x=[px],
                y=[py],
                mode="markers",
                marker=dict(size=14, color="#f1c40f", symbol="circle",
                            line=dict(width=2, color="#e67e22")),
                hoverinfo="none",
                showlegend=False,
            ))
    return traces


def _device_scatter(
    device_names: list[str],
    device_pos: dict[str, tuple[float, float]],
    device_states: dict[str, str],
) -> go.Scatter:
    """Build the device node scatter trace."""
    x = [device_pos[n][0] for n in device_names]
    y = [device_pos[n][1] for n in device_names]
    colors = [_STATE_COLORS.get(device_states.get(n, "idle"), "#95a5a6") for n in device_names]
    hover = [
        f"<b>{n}</b><br>{_STATE_LABELS.get(device_states.get(n, 'idle'), '')}"
        for n in device_names
    ]
    labels = [n.replace("_", " ") for n in device_names]

    return go.Scatter(
        x=x, y=y,
        mode="markers+text",
        marker=dict(size=30, color=colors, line=dict(width=2, color="white")),
        text=labels,
        textposition="bottom center",
        textfont=dict(size=9, color="#2c3e50"),
        hovertext=hover,
        hoverinfo="text",
        showlegend=False,
    )


def _server_scatter(server_state: str) -> go.Scatter:
    """Build the central server node scatter trace."""
    color = _STATE_COLORS.get(server_state, "#2c3e50")
    label = _STATE_LABELS.get(server_state, "")
    return go.Scatter(
        x=[_SERVER_X],
        y=[_SERVER_Y],
        mode="markers+text",
        marker=dict(size=52, color=color, symbol="diamond",
                    line=dict(width=3, color="white")),
        text=["Servidor<br>Central"],
        textposition="top center",
        textfont=dict(size=11, color="#2c3e50", family="Arial Bold"),
        hovertext=[f"<b>Servidor Central</b><br>FedAvg<br>{label}"],
        hoverinfo="text",
        showlegend=False,
    )


def build_network_figure(
    device_names: list[str],
    device_states: dict[str, str],
    packet_progress: dict[str, float],
    server_state: str = "idle",
) -> go.Figure:
    """Compose the full FL topology figure.

    Parameters
    ----------
    device_names : list[str]
        Ordered list of device identifiers.
    device_states : dict[str, str]
        Current state for each device.  Valid values: idle, ready, training,
        uploading, aggregating, done.
    packet_progress : dict[str, float]
        Upload animation progress per device in [0.0, 1.0].  A value between
        0 and 1 draws an animated yellow dot along the edge to the server.
    server_state : str
        Current state of the central server node.

    Returns
    -------
    go.Figure
        Plotly figure ready for st.plotly_chart.
    """
    positions = _circle_positions(len(device_names))
    device_pos = dict(zip(device_names, positions))

    traces: list = []
    traces.extend(_edge_traces(device_pos, device_states))
    traces.extend(_packet_traces(device_pos, packet_progress))
    traces.append(_device_scatter(device_names, device_pos, device_states))
    traces.append(_server_scatter(server_state))

    fig = go.Figure(data=traces)
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-3.8, 3.8]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-3.8, 3.8], scaleanchor="x"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=440,
    )
    return fig
