"""
Configuration module for the LunarLander-v3 PPO university project.
Defines default hyperparameters, directory paths, and reproducibility settings.
"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
VIDEOS_DIR = PROJECT_ROOT / "videos"
GRAPHS_DIR = PROJECT_ROOT / "graphs"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
REPORT_DIR = PROJECT_ROOT / "report"

# Ensure all directories exist
for directory in [MODELS_DIR, LOGS_DIR, VIDEOS_DIR, GRAPHS_DIR, SCREENSHOTS_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Hyperparameters
ENV_ID = "LunarLander-v3"
SEED = 42

# PPO Default Hyperparameters (optimized for LunarLander-v3)
DEFAULT_PPO_PARAMS = {
    "learning_rate": 3e-4,       # Step size for optimization
    "n_steps": 2048,             # Rollout buffer length per environment update
    "batch_size": 64,            # Minibatch size for gradient updates
    "n_epochs": 10,              # Number of epochs for policy optimization per update
    "gamma": 0.99,               # Discount factor for future rewards
    "gae_lambda": 0.95,          # Generalized Advantage Estimation factor
    "clip_range": 0.2,           # Clipping parameter for the policy objective
    "ent_coef": 0.01,            # Entropy coefficient to encourage exploration
    "vf_coef": 0.5,              # Value function coefficient in loss function
    "max_grad_norm": 0.5,        # Maximum gradient clipping threshold
    "verbose": 1,                # Verbosity level
}

# Training Settings
TOTAL_TIMESTEPS = 300_000
EVAL_FREQ = 20_000               # Evaluate the agent every N steps
EVAL_EPISODES = 10               # Number of evaluation episodes during training
CHECKPOINT_FREQ = 50_000         # Save model checkpoints every N steps
