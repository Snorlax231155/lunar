"""
Main orchestration script for the LunarLander-v3 PPO project.
Provides a Command-Line Interface (CLI) to run training, evaluation,
testing, hyperparameter study, video recording, and visualization.
"""

import argparse
import sys
from pathlib import Path
from src import config, utils, train, evaluate, test_agent, hyperparameter_study, visualize

def main():
    parser = argparse.ArgumentParser(
        description="University Project: LunarLander-v3 solving with PPO (Stable-Baselines3)"
    )
    
    # CLI commands
    parser.add_argument("--analyze-env", action="store_true", help="Print observation and action space details")
    parser.add_argument("--train", action="store_true", help="Train the default PPO model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the best model over 100 episodes")
    parser.add_argument("--test", action="store_true", help="Run model visually using pygame ('human' render)")
    parser.add_argument("--hyperstudy", action="store_true", help="Run hyperparameter study")
    parser.add_argument("--visualize", action="store_true", help="Generate publication-quality graphs from logs")
    parser.add_argument("--video", action="store_true", help="Record 3 episodes of best agent as MP4 files")
    parser.add_argument("--smoke-test", action="store_true", help="Run training or hyperstudy with fewer steps for fast testing")
    parser.add_argument("--all", action="store_true", help="Run entire pipeline (train, evaluate, video, hyperstudy, visualize)")
    
    # Optional arguments
    parser.add_argument("--model-path", type=str, default=None, help="Custom path to model zip file")
    parser.add_argument("--steps", type=int, default=None, help="Custom total timesteps to train")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Number of episodes for evaluation")
    parser.add_argument("--study-steps", type=int, default=100000, help="Steps per experiment in hyperparameter study")

    args = parser.parse_args()

    # If no flags are provided, print help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    if args.analyze_env:
        utils.analyze_environment()

    if args.train or args.all:
        steps = args.steps if args.steps else config.TOTAL_TIMESTEPS
        train.run_training(
            total_timesteps=steps,
            smoke_test=args.smoke_test
        )

    if args.evaluate or args.all:
        model_path = args.model_path if args.model_path else config.MODELS_DIR / "best_model.zip"
        evaluate.run_evaluation(
            model_path=model_path,
            n_episodes=args.eval_episodes
        )

    if args.video or args.all:
        model_path = args.model_path if args.model_path else config.MODELS_DIR / "best_model.zip"
        utils.record_agent_video(
            model_path=model_path,
            n_episodes=3
        )

    if args.hyperstudy or args.all:
        study_steps = 5000 if args.smoke_test else args.study_steps
        hyperparameter_study.run_study(
            steps_per_experiment=study_steps
        )

    if args.visualize or args.all:
        visualize.run_all_visualizations()

    if args.test:
        model_path = args.model_path if args.model_path else config.MODELS_DIR / "best_model.zip"
        test_agent.test_agent(
            model_path=model_path
        )

if __name__ == "__main__":
    main()
