from pathlib import Path
import sys
import time
from typing import Any, Dict

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Ensure project root is importable when running `python scripts/train_agent.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.grid_world import GridWorldEnv


class ConsoleProgressCallback(BaseCallback):
    """Logs rollout reward, episode length, and training speed to the console."""

    def __init__(self, log_interval_steps: int = 5000, verbose: int = 1) -> None:
        super().__init__(verbose=verbose)
        self.log_interval_steps: int = log_interval_steps
        self._last_log_step: int = 0
        self._start_time: float = 0.0

    def _on_training_start(self) -> None:
        self._start_time = time.perf_counter()
        self._last_log_step = 0

    def _on_step(self) -> bool:
        steps_since_log = self.num_timesteps - self._last_log_step
        if steps_since_log < self.log_interval_steps:
            return True

        self._last_log_step = self.num_timesteps
        elapsed = max(time.perf_counter() - self._start_time, 1e-6)
        fps = int(self.num_timesteps / elapsed)

        ep_buffer = getattr(self.model, "ep_info_buffer", None)
        if ep_buffer and len(ep_buffer) > 0:
            mean_reward = float(np.mean([ep["r"] for ep in ep_buffer]))
            mean_length = float(np.mean([ep["l"] for ep in ep_buffer]))
            print(
                f"[PPO Training] Steps: {self.num_timesteps:>6d} | "
                f"FPS: {fps:>4d} | "
                f"Mean Ep Reward: {mean_reward:>7.2f} | "
                f"Mean Ep Len: {mean_length:>5.1f}"
            )
        else:
            print(
                f"[PPO Training] Steps: {self.num_timesteps:>6d} | "
                f"FPS: {fps:>4d} | "
                f"Episodes completed: < 1"
            )

        return True


def get_ppo_hyperparameters() -> Dict[str, Any]:
    """Return PPO hyperparameters tuned for 8-feature 10x10 grid navigation."""
    return {
        "policy": "MlpPolicy",
        "policy_kwargs": {"net_arch": dict(pi=[128, 128], vf=[128, 128])},
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "verbose": 0,
        "seed": 42,
    }

def train(total_timesteps: int = 50_000) -> Path:
    """Train the PPO agent and persist weights to `models_weights/ppo_gridworld.zip`."""
    weights_dir = PROJECT_ROOT / "models_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    output_path = weights_dir / "ppo_gridworld.zip"

    raw_env = GridWorldEnv(grid_size=10, num_obstacles=14, max_steps=100)
    env = Monitor(raw_env)

    try:
        model = PPO(env=env, **get_ppo_hyperparameters())
        callback = ConsoleProgressCallback(log_interval_steps=5000)

        print(f"Starting PPO training for {total_timesteps:,} timesteps...")
        model.learn(total_timesteps=total_timesteps, callback=callback)
        model.save(str(output_path))
        print(f"Training complete. Model saved to: {output_path}")
    finally:
        env.close()

    return output_path


if __name__ == "__main__":
    train(total_timesteps=150_000)