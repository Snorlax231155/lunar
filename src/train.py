"""
Training script for LunarLander-v3 using PPO.
Handles training environment initialization, callback setup, model training,
and model persistence.
"""

import os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from src import config, utils, callbacks

def run_training(total_timesteps: int = config.TOTAL_TIMESTEPS,
                 seed: int = config.SEED,
                 hyperparams: dict = None,
                 tb_log_name: str = "PPO_default",
                 smoke_test: bool = False):
    """
    Sets up the environment, initializes PPO, and runs the training loop.
    
    Args:
        total_timesteps: Number of environment steps to train for.
        seed: Random seed.
        hyperparams: Dictionary of PPO hyperparameters to override defaults.
        tb_log_name: Subfolder name for TensorBoard logs.
        smoke_test: If True, overrides steps to 5000 for rapid debugging.
    """
    # Seeding
    utils.set_seed(seed)
    
    if smoke_test:
        total_timesteps = 5000
        print(f"Smoke test enabled! Training for only {total_timesteps} steps...")

    # Create train and evaluation environments
    train_env = utils.make_env(env_id=config.ENV_ID, seed=seed)
    train_env = Monitor(train_env, filename=str(config.LOGS_DIR / f"{tb_log_name}_train"))
    
    eval_env = utils.make_env(env_id=config.ENV_ID, seed=seed + 1)
    eval_env = Monitor(eval_env, filename=str(config.LOGS_DIR / f"{tb_log_name}_eval"))

    # Prepare hyperparameters
    params = config.DEFAULT_PPO_PARAMS.copy()
    if hyperparams:
        params.update(hyperparams)
        
    print("=" * 60)
    print(f"Starting PPO training on {config.ENV_ID} for {total_timesteps} steps.")
    print(f"Seed: {seed}")
    print("Hyperparameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Initialize model
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        tensorboard_log=str(config.LOGS_DIR),
        seed=seed,
        device="cpu",
        **params
    )

    # Custom metrics callback
    metrics_callback = callbacks.LunarLanderMetricsCallback(window_size=50, verbose=1)

    # EvalCallback for saving the best model based on evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(config.MODELS_DIR),
        log_path=str(config.LOGS_DIR),
        eval_freq=max(500, config.EVAL_FREQ // 4) if smoke_test else config.EVAL_FREQ,
        n_eval_episodes=config.EVAL_EPISODES,
        deterministic=True,
        render=False
    )

    callback_list = CallbackList([metrics_callback, eval_callback])

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        tb_log_name=tb_log_name
    )

    # Save final model
    final_model_path = config.MODELS_DIR / f"{tb_log_name}_final.zip"
    model.save(final_model_path)
    print(f"Training completed successfully!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {config.MODELS_DIR / 'best_model.zip'}")

    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test")
    args = parser.parse_args()
    
    run_training(smoke_test=args.smoke_test)
