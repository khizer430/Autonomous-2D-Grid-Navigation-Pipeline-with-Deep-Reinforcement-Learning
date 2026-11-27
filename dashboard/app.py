from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8000"


def call_simulate_endpoint(
    api_url: str,
    seed: Optional[int],
    deterministic: bool,
    max_steps: int,
) -> Dict[str, Any]:
    """Send a simulation request to the FastAPI backend and return parsed JSON."""
    payload: Dict[str, Any] = {
        "seed": seed,
        "deterministic": deterministic,
        "max_steps": max_steps,
    }
    response = requests.post(
        f"{api_url.rstrip('/')}/simulate",
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def build_navigation_figure(sim_data: Dict[str, Any]) -> go.Figure:
    """Build an animated 2D Plotly figure showing obstacles, goal, path, and agent."""
    grid_size: int = int(sim_data["grid_size"])
    path: List[List[int]] = sim_data["path"]
    goal: List[int] = sim_data["goal"]
    obstacles: List[List[int]] = sim_data["obstacles"]

    def make_traces_for_step(step_idx: int) -> List[Any]:
        sub_path = path[: step_idx + 1]
        path_x = [pt[0] for pt in sub_path]
        path_y = [pt[1] for pt in sub_path]
        agent_x, agent_y = path[step_idx][0], path[step_idx][1]

        traces: List[Any] = []

        # 1. Static obstacles (black squares)
        if obstacles:
            traces.append(
                go.Scatter(
                    x=[obs[0] for obs in obstacles],
                    y=[obs[1] for obs in obstacles],
                    mode="markers",
                    name="Obstacles",
                    marker=dict(
                        symbol="square",
                        size=28,
                        color="#111827",
                        line=dict(color="#000000", width=1),
                    ),
                    hovertemplate="Obstacle: (%{x}, %{y})<extra></extra>",
                )
            )

        # 2. Goal cell (green marker)
        traces.append(
            go.Scatter(
                x=[goal[0]],
                y=[goal[1]],
                mode="markers",
                name="Goal",
                marker=dict(
                    symbol="square",
                    size=30,
                    color="#16A34A",
                    line=dict(color="#14532D", width=2),
                ),
                hovertemplate="Goal: (%{x}, %{y})<extra></extra>",
            )
        )

        # 3. Trajectory path so far (red line)
        traces.append(
            go.Scatter(
                x=path_x,
                y=path_y,
                mode="lines+markers",
                name="Path",
                line=dict(color="#DC2626", width=3),
                marker=dict(size=5, color="#EF4444"),
                hovertemplate="Step Path: (%{x}, %{y})<extra></extra>",
            )
        )

        # 4. Current agent position (highlighted red circle)
        traces.append(
            go.Scatter(
                x=[agent_x],
                y=[agent_y],
                mode="markers",
                name="Agent",
                marker=dict(
                    symbol="circle",
                    size=16,
                    color="#B91C1C",
                    line=dict(color="#FFFFFF", width=2),
                ),
                hovertemplate=f"Step {step_idx}: ({agent_x}, {agent_y})<extra></extra>",
            )
        )

        return traces

    initial_traces = make_traces_for_step(0)

    frames: List[go.Frame] = [
        go.Frame(
            name=f"step_{idx}",
            data=make_traces_for_step(idx),
            layout=go.Layout(title=f"rl-nav Trajectory — Step {idx} / {len(path) - 1}"),
        )
        for idx in range(len(path))
    ]

    fig = go.Figure(data=initial_traces, frames=frames)

    fig.update_layout(
        title=f"rl-nav Trajectory — Step 0 / {max(len(path) - 1, 0)}",
        width=640,
        height=660,
        margin=dict(t=115, r=120, b=60, l=60),
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(
            range=[-0.5, grid_size - 0.5],
            dtick=1,
            title="X Coordinate",
            showgrid=True,
            gridcolor="#CBD5E1",
            zeroline=False,
        ),
        yaxis=dict(
            range=[grid_size - 0.5, -0.5],  # Top-down grid matching (0,0) top-left
            dtick=1,
            title="Y Coordinate",
            showgrid=True,
            gridcolor="#CBD5E1",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.0,
                y=1.08,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=120, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=40),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.20,
                y=1.08,
                len=0.80,
                xanchor="left",
                yanchor="top",
                pad=dict(t=0, b=0),
                currentvalue=dict(prefix="Step: ", visible=True),
                steps=[
                    dict(
                        label=str(idx),
                        method="animate",
                        args=[
                            [f"step_{idx}"],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    )
                    for idx in range(len(path))
                ],
            )
        ],
    )

    return fig


def main() -> None:
    """Render the Streamlit dashboard for rl-nav."""
    st.set_page_config(page_title="rl-nav Dashboard", layout="wide")
    st.title("rl-nav: RL Grid-World Navigation Dashboard")
    st.caption(
        "Animate rollouts returned by the FastAPI `/simulate` endpoint over the 10x10 GridWorld."
    )

    with st.sidebar:
        st.header("Simulation Controls")
        api_url = st.text_input("FastAPI Base URL", value=DEFAULT_API_URL)
        use_seed = st.checkbox("Use Fixed Seed", value=True)
        seed: Optional[int] = None
        if use_seed:
            seed = st.number_input(
                "Seed", min_value=0, max_value=1_000_000, value=42, step=1
            )
        deterministic = st.checkbox("Deterministic Policy", value=True)
        max_steps = st.slider(
            "Max Steps", min_value=10, max_value=200, value=100, step=10
        )
        run_clicked = st.button("Run Simulation", type="primary", use_container_width=True)

    if "sim_result" not in st.session_state:
        st.session_state.sim_result = None

    if run_clicked:
        try:
            with st.spinner("Running episode via FastAPI backend..."):
                st.session_state.sim_result = call_simulate_endpoint(
                    api_url=api_url,
                    seed=int(seed) if seed is not None else None,
                    deterministic=deterministic,
                    max_steps=max_steps,
                )
        except requests.RequestException as exc:
            st.error(f"Failed to reach FastAPI backend at {api_url}: {exc}")

    sim_data: Optional[Dict[str, Any]] = st.session_state.sim_result
    if sim_data is None:
        st.info('Click **"Run Simulation"** in the sidebar to fetch and animate a rollout.')
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", str(sim_data.get("status", "unknown")).upper())
    col2.metric("Reached Goal", "Yes" if sim_data.get("reached_goal") else "No")
    col3.metric("Steps Taken", int(sim_data.get("steps", 0)))
    col4.metric("Total Reward", float(sim_data.get("total_reward", 0.0)))

    chart_col, log_col = st.columns([1.35, 1.0])

    with chart_col:
        fig = build_navigation_figure(sim_data)
        st.plotly_chart(fig, use_container_width=True)

    with log_col:
        st.subheader("Episode Coordinates")
        st.write(f"**Start:** `{sim_data['start']}` | **Goal:** `{sim_data['goal']}`")
        path_df = pd.DataFrame(sim_data["path"], columns=["x", "y"])
        path_df.index.name = "step"
        st.dataframe(path_df, height=460, use_container_width=True)


if __name__ == "__main__":
    main()