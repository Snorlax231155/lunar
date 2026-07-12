"""
Demonstration script to run a trained LunarLander-v3 agent.
Renders the environment using human mode to visually inspect the agent's behavior.
"""

import os
from pathlib import Path
from stable_baselines3 import PPO
from src import config, utils

def test_agent(model_path: str = None, n_episodes: int = 5, seed: int = config.SEED + 200):
    """
    Runs the trained PPO agent on the LunarLander-v3 environment with render_mode='human'.
    
    Args:
        model_path: Path to the trained model.
        n_episodes: Number of episodes to run.
        seed: Random seed.
    """
    if model_path is None:
        model_path = config.MODELS_DIR / "best_model.zip"
    else:
        model_path = Path(model_path)
        
    if not os.path.exists(model_path):
        model_path = config.MODELS_DIR / "PPO_default_final.zip"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model found at {model_path} or best_model.zip")
            
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    
    # Human mode (uses pygame window)
    render_mode = "human"
    
    print(f"Starting visual test with render_mode='{render_mode}'...")
    try:
        env = utils.make_env(env_id=config.ENV_ID, seed=seed, render_mode=render_mode)
    except Exception as e:
        print(f"\n[Warning] Failed to initialize environment in '{render_mode}' render mode.")
        print(f"Error: {e}")
        print("This is expected if you are running in a headless environment without an X server.")
        print("If you wish to view gameplay, please check the MP4 videos recorded in the 'videos/' directory.")
        return

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
        print(f"Episode {ep + 1}/{n_episodes} | Total Steps: {steps} | Cumulative Reward: {total_reward:.2f}")
        
    env.close()
    print("Agent testing completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to model zip file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to test")
    args = parser.parse_args()
    
    test_agent(model_path=args.model, n_episodes=args.episodes)
