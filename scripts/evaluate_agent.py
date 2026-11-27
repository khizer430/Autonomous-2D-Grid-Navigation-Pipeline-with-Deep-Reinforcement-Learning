from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

# Ensure project root is importable when running `python scripts/evaluate_agent.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.grid_world import GridWorldEnv


def evaluate_policy(
    model: PPO,
    num_episodes: int = 100,
    base_seed: int = 1000,
) -> Dict[str, Any]:
    """Run `num_episodes` evaluation rollouts and compute navigation metrics."""
    env = GridWorldEnv(grid_size=10, num_obstacles=14, max_steps=100, render_mode=None)

    successes: int = 0
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    successful_lengths: List[int] = []

    try:
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=base_seed + ep)
            terminated: bool = False
            truncated: bool = False
            total_reward: float = 0.0
            steps: int = 0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total_reward += float(reward)
                steps += 1

            reached_goal = bool(info.get("reached_goal", False))
            if reached_goal:
                successes += 1
                successful_lengths.append(steps)

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
    finally:
        env.close()

    success_rate = (successes / num_episodes) * 100.0
    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    std_reward = float(np.std(episode_rewards)) if episode_rewards else 0.0
    mean_steps = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    mean_success_steps = (
        float(np.mean(successful_lengths)) if successful_lengths else float("nan")
    )

    return {
        "episodes": num_episodes,
        "successes": successes,
        "success_rate_pct": success_rate,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_steps": mean_steps,
        "mean_success_steps": mean_success_steps,
    }


def record_rollout_gif(
    model: PPO,
    output_gif: Path,
    seed: int = 1000,
    fps: int = 8,
) -> Tuple[bool, int]:
    """Record a single episode as an animated GIF and return (reached_goal, steps)."""
    env = GridWorldEnv(
        grid_size=10, num_obstacles=14, max_steps=100, render_mode="rgb_array"
    )
    frames: List[np.ndarray] = []
    steps: int = 0
    reached_goal: bool = False

    try:
        obs, _ = env.reset(seed=seed)
        first_frame = env.render()
        if first_frame is not None:
            frames.append(first_frame)

        terminated: bool = False
        truncated: bool = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            steps += 1
            reached_goal = bool(info.get("reached_goal", False))

            frame = env.render()
            if frame is not None:
                frames.append(frame)
    finally:
        env.close()

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / max(fps, 1))
    imageio.mimsave(output_gif, frames, duration=duration_ms, loop=0)

    return reached_goal, steps


def main() -> None:
    """Load trained PPO weights, run 100-episode evaluation, and save rollout GIF."""
    model_path = PROJECT_ROOT / "models_weights" / "ppo_gridworld.zip"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run scripts/train_agent.py first."
        )

    model = PPO.load(str(model_path))
    metrics = evaluate_policy(model=model, num_episodes=100, base_seed=1000)

    print("=== Phase 3 Evaluation Results (100 Episodes) ===")
    print(f"Successes         : {metrics['successes']} / {metrics['episodes']}")
    print(f"Success Rate      : {metrics['success_rate_pct']:.2f}%")
    print(
        f"Mean Episode Rew  : {metrics['mean_reward']:.2f} "
        f"(+/- {metrics['std_reward']:.2f})"
    )
    print(f"Mean Episode Steps: {metrics['mean_steps']:.1f}")
    if not np.isnan(metrics["mean_success_steps"]):
        print(f"Mean Success Steps: {metrics['mean_success_steps']:.1f}")
    else:
        print("Mean Success Steps: N/A (0 successful episodes)")

    gif_path = PROJECT_ROOT / "outputs" / "rl-nav_rollout.gif"
    gif_goal, gif_steps = record_rollout_gif(model=model, output_gif=gif_path, seed=1000)
    print(
        f"Saved rollout GIF to: {gif_path} "
        f"(Goal reached: {gif_goal}, Steps: {gif_steps})"
    )


if __name__ == "__main__":
    main()