# Autonomous 2D Grid Navigation Pipeline with Deep Reinforcement Learning (`rl-nav`)

An end-to-end 2D grid-world navigation system built from scratch using a custom **OpenAI Gymnasium** environment, **Proximal Policy Optimization (PPO)** in **Stable-Baselines3 (PyTorch)**, a **FastAPI** simulation service, and an interactive **Streamlit + Plotly** dashboard.

---

## Architecture Overview

```text
+-----------------------+       +-------------------------------+
|  Streamlit Dashboard  | ----> |  FastAPI Backend (/simulate)  |
|  (Plotly Step Replay) | <---- |  (Cached PPO Policy Inference)|
+-----------------------+       +---------------+---------------+
                                                |
                                                v
                                +-------------------------------+
                                |  Gymnasium GridWorldEnv       |
                                |  (10x10 Grid, BFS Solvability,|
                                |   Pygame & RGB Frame Render)  |
                                +---------------+---------------+
                                                ^
                                                |
                                +---------------+---------------+
                                |  PPO Training & Evaluation    |
                                |  (Stable-Baselines3 / PyTorch)|
                                +-------------------------------+

Repo Structure:
Autonomous-2D-Grid-Navigation-Pipeline-with-Deep-Reinforcement-Learning/
├── dashboard/
│   └── app.py                  # Streamlit + Plotly interactive trajectory visualizer
├── models_weights/             # Saved SB3 model weights (*.zip, gitignored by default)
├── outputs/                    # Generated evaluation rollouts (*.gif, gitignored by default)
├── scripts/
│   ├── train_agent.py          # PPO training script with console progress callback
│   └── evaluate_agent.py       # 100-episode evaluation & GIF generator
├── src/
│   ├── env/
│   │   └── grid_world.py       # Custom Gymnasium GridWorldEnv
│   └── serving/
│       └── api.py              # FastAPI backend with /health and /simulate endpoints
├── requirements.txt
└── README.md

Setup & Usage (Windows PowerShell)
1. Create Environment & Install Dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install imageio>=2.34.0 pillow>=10.2.0

2. Train the PPO Agent:
python scripts\train_agent.py

3.Evaluate Over 100 Episodes & Generate GIF:
python scripts\evaluate_agent.py

4. Run FastAPI Backend & Streamlit Dashboard
Terminal 1 (FastAPI server):
uvicorn src.serving.api:app --reload --host 127.0.0.1 --port 8000

Terminal 2 (Streamlit dashboard):
streamlit run dashboard\app.py
