"""
Utility functions for the LunarLander-v3 project.
Includes environment creation with wrappers, seeding, and space analysis tools.
"""

import os
import random
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3.common.monitor import Monitor
from src import config

def set_seed(seed: int = 42):
    """
    Sets seed for reproducibility across random, numpy, torch, and gymnasium.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Configure PyTorch deterministic behaviors
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_env(env_id: str = config.ENV_ID,
             seed: int = config.SEED,
             render_mode: str = None,
             record_video_dir: str = None,
             episode_trigger = None):
    """
    Creates a Gymnasium environment wrapped with Monitor, RecordEpisodeStatistics, and optionally RecordVideo.
    """
    env = gym.make(env_id, render_mode=render_mode)
    
    # Reset seed
    env.reset(seed=seed)
    
    # Record Statistics
    env = RecordEpisodeStatistics(env)
    
    # Record Video wrapper if directory is provided
    if record_video_dir is not None:
        if episode_trigger is None:
            # Record every episode
            episode_trigger = lambda ep: True
        env = RecordVideo(
            env,
            video_folder=record_video_dir,
            episode_trigger=episode_trigger,
            name_prefix="lunar_lander_vid"
        )
        
    return env

def record_agent_video(model_path: str = None, n_episodes: int = 3, seed: int = config.SEED + 300):
    """
    Evaluates a trained model and records video files of the episodes in videos/ folder.
    """
    from stable_baselines3 import PPO
    
    if model_path is None:
        model_path = config.MODELS_DIR / "best_model.zip"
    else:
        model_path = Path(model_path)
        
    if not os.path.exists(model_path):
        model_path = config.MODELS_DIR / "PPO_default_final.zip"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model found at {model_path} or best_model.zip")
            
    print(f"Recording {n_episodes} episodes using model: {model_path}")
    model = PPO.load(model_path)
    
    # Use rgb_array for headless video recording
    env = make_env(
        env_id=config.ENV_ID,
        seed=seed,
        render_mode="rgb_array",
        record_video_dir=str(config.VIDEOS_DIR),
        episode_trigger=lambda ep: True
    )
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
        print(f"  Recorded Episode {ep + 1} | Reward: {ep_reward:.2f}")
            
    env.close()
    print(f"Videos successfully recorded in: {config.VIDEOS_DIR}")

def analyze_environment():
    """
    Programmatically analyzes and prints the LunarLander-v3 observation and action spaces.
    """
    env = gym.make(config.ENV_ID)
    obs_space = env.observation_space
    act_space = env.action_space
    
    print("=" * 60)
    print("           LUNARLANDER ENVIRONMENT ANALYSIS")
    print("=" * 60)
    print(f"Environment ID: {config.ENV_ID}")
    print(f"Observation Space: {obs_space}")
    print(f"Observation Low:  {obs_space.low}")
    print(f"Observation High: {obs_space.high}")
    print(f"Action Space:      {act_space}")
    print("-" * 60)
    
    obs_explanations = [
        "1. Horizontal Position (x): Coordinates of lander center of mass (limits: [-1.5, 1.5] approx for landing pad).",
        "2. Vertical Position (y): Coordinates of lander center of mass.",
        "3. Horizontal Velocity (vx): Linear velocity in the horizontal axis.",
        "4. Vertical Velocity (vy): Linear velocity in the vertical axis.",
        "5. Lander Angle (theta): Orientation angle of the lander (0 is vertical).",
        "6. Angular Velocity (dtheta): Rate of change of the lander orientation angle.",
        "7. Left Leg Contact: Boolean/float (1.0 if left landing leg touches ground, 0.0 otherwise).",
        "8. Right Leg Contact: Boolean/float (1.0 if right landing leg touches ground, 0.0 otherwise)."
    ]
    
    print("Observation Variables:")
    for line in obs_explanations:
        print(f"  {line}")
        
    print("-" * 60)
    action_explanations = [
        "0: Do nothing (Free fall under moon gravity).",
        "1: Fire left orientation engine (Rotates lander clockwise).",
        "2: Fire main engine (Provides upward thrust, consumes fuel).",
        "3: Fire right orientation engine (Rotates lander counter-clockwise)."
    ]
    print("Action Space (Discrete(4)):")
    for line in action_explanations:
        print(f"  {line}")
    print("=" * 60)
    env.close()

if __name__ == "__main__":
    analyze_environment()
