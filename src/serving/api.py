from contextlib import asynccontextmanager
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from stable_baselines3 import PPO

# Ensure project root is importable when running uvicorn from repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.grid_world import GridWorldEnv

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models_weights" / "ppo_gridworld.zip"


class SimulateRequest(BaseModel):
    """Optional parameters for running a single episode simulation."""

    seed: Optional[int] = Field(
        default=None, description="Optional RNG seed for reproducible episode layout."
    )
    deterministic: bool = Field(
        default=True, description="Whether to select policy actions deterministically."
    )
    max_steps: int = Field(
        default=100, ge=1, le=500, description="Maximum steps allowed in the episode."
    )


class SimulateResponse(BaseModel):
    """Trajectory and environment metadata returned after simulating one episode."""

    status: str
    path: List[List[int]]
    start: List[int]
    goal: List[int]
    obstacles: List[List[int]]
    grid_size: int
    steps: int
    reached_goal: bool
    total_reward: float


def run_single_episode(
    model: PPO,
    seed: Optional[int] = None,
    deterministic: bool = True,
    max_steps: int = 100,
) -> Dict[str, Any]:
    """Simulate one episode with the trained PPO agent and return trajectory data."""
    env = GridWorldEnv(
        grid_size=10, num_obstacles=14, max_steps=max_steps, render_mode=None
    )

    try:
        obs, info = env.reset(seed=seed)
        current_pos: Tuple[int, int] = (int(obs[0]), int(obs[1]))
        path: List[List[int]] = [[current_pos[0], current_pos[1]]]

        terminated: bool = False
        truncated: bool = False
        total_reward: float = 0.0
        steps: int = 0
        last_info: Dict[str, Any] = info

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, last_info = env.step(int(action))
            total_reward += float(reward)
            steps += 1
            path.append([int(obs[0]), int(obs[1])])

        start_coord = last_info["start"]
        goal_coord = last_info["goal"]
        obstacles_coords = last_info["obstacles"]
        reached_goal = bool(last_info.get("reached_goal", False))

        return {
            "status": "success" if reached_goal else "incomplete",
            "path": path,
            "start": [int(start_coord[0]), int(start_coord[1])],
            "goal": [int(goal_coord[0]), int(goal_coord[1])],
            "obstacles": [[int(ox), int(oy)] for ox, oy in obstacles_coords],
            "grid_size": env.grid_size,
            "steps": steps,
            "reached_goal": reached_goal,
            "total_reward": round(total_reward, 2),
        }
    finally:
        env.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained PPO model once at application startup."""
    if not DEFAULT_MODEL_PATH.exists():
        raise RuntimeError(
            f"Trained model weights not found at {DEFAULT_MODEL_PATH}. "
            "Run scripts/train_agent.py first."
        )
    app.state.model = PPO.load(str(DEFAULT_MODEL_PATH))
    app.state.model_path = str(DEFAULT_MODEL_PATH)
    yield
    app.state.model = None


app = FastAPI(
    title="rl-nav Simulation API",
    description="FastAPI backend serving trained RL navigation rollouts for GridWorldEnv.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Verify service health and model availability."""
    model_loaded = getattr(app.state, "model", None) is not None
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_path": getattr(app.state, "model_path", None),
    }


@app.post("/simulate", response_model=SimulateResponse)
def simulate_post(request: Optional[SimulateRequest] = None) -> SimulateResponse:
    """Simulate a single episode and return the trajectory path."""
    model: Optional[PPO] = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="RL model is not loaded.")

    payload = request or SimulateRequest()
    result = run_single_episode(
        model=model,
        seed=payload.seed,
        deterministic=payload.deterministic,
        max_steps=payload.max_steps,
    )
    return SimulateResponse(**result)


@app.get("/simulate", response_model=SimulateResponse)
def simulate_get(
    seed: Optional[int] = None,
    deterministic: bool = True,
    max_steps: int = 100,
) -> SimulateResponse:
    """Convenience GET endpoint to simulate a single episode."""
    model: Optional[PPO] = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="RL model is not loaded.")

    result = run_single_episode(
        model=model,
        seed=seed,
        deterministic=deterministic,
        max_steps=max_steps,
    )
    return SimulateResponse(**result)