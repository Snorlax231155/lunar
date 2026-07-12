"""
Visualization module for generating publication-quality figures.
Plots training curves, evaluation metrics, and hyperparameter comparisons.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import config

# Set style for publication quality
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'legend.fontsize': 10,
    'grid.alpha': 0.3,
    'savefig.bbox': 'tight',
    'savefig.dpi': 300
})

def plot_training_curves(monitor_path: str = None, save_path: str = None):
    """
    Plots the training rewards and episode lengths with moving averages.
    """
    if monitor_path is None:
        monitor_path = config.LOGS_DIR / "PPO_default_train.monitor.csv"
    else:
        monitor_path = Path(monitor_path)
        
    if save_path is None:
        save_path = config.GRAPHS_DIR / "training_curves.png"
    else:
        save_path = Path(save_path)

    if not monitor_path.exists():
        print(f"[Warning] Monitor file {monitor_path} not found. Skipping plot.")
        return

    # Load data
    df = pd.read_csv(monitor_path, skiprows=1)
    
    # Calculate rolling metrics
    window = min(50, len(df))
    df['rolling_r'] = df['r'].rolling(window=window, min_periods=1).mean()
    df['rolling_l'] = df['l'].rolling(window=window, min_periods=1).mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Reward plot
    ax1.plot(df.index, df['r'], color='skyblue', alpha=0.4, label='Raw Episode Reward')
    ax1.plot(df.index, df['rolling_r'], color='navy', linewidth=2, label=f'{window}-Ep Rolling Mean')
    ax1.axhline(y=200, color='green', linestyle='--', label='Solving Threshold (200)')
    ax1.set_title('Training Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # Length plot
    ax2.plot(df.index, df['l'], color='salmon', alpha=0.4, label='Raw Episode Length')
    ax2.plot(df.index, df['rolling_l'], color='crimson', linewidth=2, label=f'{window}-Ep Rolling Mean')
    ax2.set_title('Training Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to: {save_path}")

def plot_evaluation_progress(eval_npz_path: str = None, save_path: str = None):
    """
    Plots evaluation mean reward vs. training steps with a shaded standard deviation band.
    """
    if eval_npz_path is None:
        eval_npz_path = config.LOGS_DIR / "evaluations.npz"
    else:
        eval_npz_path = Path(eval_npz_path)
        
    if save_path is None:
        save_path = config.GRAPHS_DIR / "evaluation_progress.png"
    else:
        save_path = Path(save_path)

    if not eval_npz_path.exists():
        print(f"[Warning] Evaluations file {eval_npz_path} not found. Skipping plot.")
        return

    data = np.load(eval_npz_path)
    timesteps = data['timesteps']
    results = data['results']  # Shape: (n_evals, n_episodes)
    
    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)
    
    plt.figure(figsize=(9, 5))
    plt.plot(timesteps, mean_rewards, color='blue', marker='o', label='Mean Eval Reward')
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, color='blue', alpha=0.15, label='Std Dev')
    plt.axhline(y=200, color='green', linestyle='--', label='Solving Threshold (200)')
    
    plt.title('Evaluation Reward Progress')
    plt.xlabel('Training Steps')
    plt.ylabel('Evaluation Reward (10 Episodes)')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Evaluation progress plot saved to: {save_path}")

def plot_hyperparameter_comparison(study_csv_path: str = None, save_path: str = None):
    """
    Creates a bar plot comparing the performance of different hyperparameters.
    """
    if study_csv_path is None:
        study_csv_path = config.REPORT_DIR / "hyperparameter_study_results.csv"
    else:
        study_csv_path = Path(study_csv_path)
        
    if save_path is None:
        save_path = config.GRAPHS_DIR / "hyperparameter_comparison.png"
    else:
        save_path = Path(save_path)

    if not study_csv_path.exists():
        print(f"[Warning] Hyperparameter study CSV {study_csv_path} not found. Skipping plot.")
        return

    df = pd.read_csv(study_csv_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Colors
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(df)))
    
    # Reward comparison
    ax1.bar(df['Experiment'], df['Avg Reward'], color=colors, edgecolor='black', alpha=0.8)
    ax1.axhline(y=200, color='red', linestyle='--', label='Solved (200)')
    ax1.set_title('Average Evaluation Reward')
    ax1.set_ylabel('Reward')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['Experiment'], rotation=45, ha='right')
    ax1.grid(True, axis='y')
    ax1.legend(loc='lower right')
    
    # Success Rate comparison
    ax2.bar(df['Experiment'], df['Success Rate (%)'], color=colors, edgecolor='black', alpha=0.8)
    ax2.set_title('Landing Success Rate')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['Experiment'], rotation=45, ha='right')
    ax2.set_ylim(0, 110)
    ax2.grid(True, axis='y')
    
    plt.suptitle('Hyperparameter Performance Comparison (100k Steps Study)', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Hyperparameter comparison plot saved to: {save_path}")

def plot_hyperparameter_learning_curves(save_path: str = None):
    """
    Plots the moving average training curves of all hyperparameter study experiments on a single graph.
    """
    if save_path is None:
        save_path = config.GRAPHS_DIR / "hyperparameter_learning_curves.png"
    else:
        save_path = Path(save_path)
        
    experiments = [
        "Baseline", "LR_1e-4", "LR_1e-3", "Gamma_0.95", "Gamma_0.999", "Clip_0.1", "Ent_0.05"
    ]
    
    plt.figure(figsize=(10, 6))
    
    found_any = False
    for name in experiments:
        monitor_path = config.LOGS_DIR / f"study_{name}_train.monitor.csv"
        if monitor_path.exists():
            df = pd.read_csv(monitor_path, skiprows=1)
            window = min(100, len(df))
            rolling_r = df['r'].rolling(window=window, min_periods=1).mean()
            steps = df['l'].cumsum()
            
            plt.plot(steps, rolling_r, label=name, linewidth=2)
            found_any = True
            
    if not found_any:
        print("[Warning] No hyperparameter study monitor files found. Skipping plot.")
        plt.close()
        return
        
    plt.axhline(y=200, color='black', linestyle='--', label='Solving Threshold (200)')
    plt.title('Training Curves Across Hyperparameters')
    plt.xlabel('Training Steps')
    plt.ylabel('Rolling Mean Reward (100 Episodes)')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Hyperparameter learning curves saved to: {save_path}")

def run_all_visualizations():
    """
    Runs all visualization functions.
    """
    plot_training_curves()
    plot_evaluation_progress()
    plot_hyperparameter_comparison()
    plot_hyperparameter_learning_curves()

if __name__ == "__main__":
    run_all_visualizations()
