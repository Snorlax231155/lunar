"""
Evaluation script for trained PPO models on LunarLander-v3.
Loads a model and evaluates it over a specified number of episodes,
calculating mean/min/max rewards, success/crash rates, and episode lengths.
"""

import json
import os
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from src import config, utils

def run_evaluation(model_path: str = None, n_episodes: int = 100, seed: int = config.SEED + 100):
    """
    Evaluates a trained model over a set number of episodes and prints key metrics.
    
    Args:
        model_path: Path to the zip model. Defaults to models/best_model.zip.
        n_episodes: Number of episodes to evaluate.
        seed: Seed for the environment.
    """
    if model_path is None:
        model_path = config.MODELS_DIR / "best_model.zip"
    else:
        model_path = Path(model_path)
        
    if not os.path.exists(model_path):
        # Fall back to final model if best model doesn't exist
        model_path = config.MODELS_DIR / "PPO_default_final.zip"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model found at {model_path} or best_model.zip")
            
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create evaluation environment
    env = utils.make_env(env_id=config.ENV_ID, seed=seed, render_mode=None)
    
    # Run evaluation
    episode_rewards = []
    episode_lengths = []
    successes = []
    crashes = []
    
    print(f"Evaluating model over {n_episodes} episodes...")
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        ep_len = 0
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        
        # Check landing outcomes
        is_success = ep_reward >= 200.0
        is_crash = ep_reward <= -100.0
        
        successes.append(1.0 if is_success else 0.0)
        crashes.append(1.0 if is_crash else 0.0)
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{n_episodes} | Reward: {ep_reward:.2f} | Length: {ep_len}")
            
    # Calculate statistics
    avg_reward = float(np.mean(episode_rewards))
    max_reward = float(np.max(episode_rewards))
    min_reward = float(np.min(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    success_rate = float(np.mean(successes)) * 100.0
    crash_rate = float(np.mean(crashes)) * 100.0
    avg_length = float(np.mean(episode_lengths))
    
    # Print results in a clean table
    print("\n" + "=" * 60)
    print("                 EVALUATION METRICS SUMMARY")
    print("=" * 60)
    print(f"Model Evaluated:            {model_path.name}")
    print(f"Total Episodes:             {n_episodes}")
    print(f"Average Episode Length:     {avg_length:.1f} steps")
    print(f"Average Reward (Score):     {avg_reward:.2f}")
    print(f"Max Reward:                 {max_reward:.2f}")
    print(f"Min Reward:                 {min_reward:.2f}")
    print(f"Standard Deviation:         {std_reward:.2f}")
    print(f"Landing Success Rate:       {success_rate:.1f}%")
    print(f"Crash Rate:                 {crash_rate:.1f}%")
    print("=" * 60)
    
    # Save evaluation metrics to json
    results = {
        "model_name": model_path.name,
        "n_episodes": n_episodes,
        "avg_reward": avg_reward,
        "max_reward": max_reward,
        "min_reward": min_reward,
        "std_reward": std_reward,
        "success_rate": success_rate,
        "crash_rate": crash_rate,
        "avg_length": avg_length
    }
    
    results_path = config.REPORT_DIR / f"{model_path.stem}_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to: {results_path}")
    env.close()
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to model zip file")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    args = parser.parse_args()
    
    run_evaluation(model_path=args.model, n_episodes=args.episodes)
