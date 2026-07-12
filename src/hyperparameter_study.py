"""
Hyperparameter study module.
Runs comparative training runs with different hyperparameters (learning rate,
gamma, clip range, entropy coefficient) and logs performance metrics.
"""

import os
import time
import pandas as pd
import numpy as np
from src import config, train, evaluate

def run_study(steps_per_experiment: int = 100000, seed: int = config.SEED):
    """
    Runs PPO training for various hyperparameter settings and evaluates each.
    """
    experiments = [
        # (Name, Parameter dict)
        ("Baseline", {}),
        ("LR_1e-4", {"learning_rate": 1e-4}),
        ("LR_1e-3", {"learning_rate": 1e-3}),
        ("Gamma_0.95", {"gamma": 0.95}),
        ("Gamma_0.999", {"gamma": 0.999}),
        ("Clip_0.1", {"clip_range": 0.1}),
        ("Ent_0.05", {"ent_coef": 0.05}),
    ]

    results = []

    print("=" * 60)
    print(f"STARTING HYPERPARAMETER STUDY ({len(experiments)} Experiments)")
    print(f"Steps per experiment: {steps_per_experiment}")
    print("=" * 60)

    for name, overrides in experiments:
        tb_log_name = f"study_{name}"
        model_path = config.MODELS_DIR / f"{tb_log_name}_final.zip"
        
        print(f"\n[Hyperstudy] Running experiment: {name} ...")
        
        # Measure training time
        start_time = time.time()
        train.run_training(
            total_timesteps=steps_per_experiment,
            seed=seed,
            hyperparams=overrides,
            tb_log_name=tb_log_name
        )
        train_time = time.time() - start_time
        
        # Evaluate model
        print(f"[Hyperstudy] Evaluating model {name}...")
        eval_metrics = evaluate.run_evaluation(
            model_path=model_path,
            n_episodes=30,
            seed=seed + 500
        )
        
        # Log results
        run_data = {
            "Experiment": name,
            "Train Time (s)": round(train_time, 2),
            "Avg Reward": round(eval_metrics["avg_reward"], 2),
            "Max Reward": round(eval_metrics["max_reward"], 2),
            "Std Dev Reward": round(eval_metrics["std_reward"], 2),
            "Success Rate (%)": round(eval_metrics["success_rate"], 1),
            "Crash Rate (%)": round(eval_metrics["crash_rate"], 1),
            "Avg Ep Length": round(eval_metrics["avg_length"], 1)
        }
        results.append(run_data)
        
    # Create DataFrame and save
    df = pd.DataFrame(results)
    csv_path = config.REPORT_DIR / "hyperparameter_study_results.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 60)
    print("             HYPERPARAMETER STUDY RESULTS SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    print(f"Results saved to: {csv_path}")
    
    return csv_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100000, help="Steps per experiment")
    args = parser.parse_args()
    
    run_study(steps_per_experiment=args.steps)
