"""
Custom callbacks for Stable-Baselines3 training.
Tracks running success/crash rates, average episode length, and rewards,
logging them to TensorBoard for easy monitoring.
"""

from collections import deque
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class LunarLanderMetricsCallback(BaseCallback):
    """
    Callback for tracking and logging custom LunarLander metrics:
    - Success Rate (rollout average of episodes with reward >= 200)
    - Crash Rate (rollout average of episodes with reward <= -100)
    - Rolling Mean Reward
    - Rolling Mean Episode Length
    """
    def __init__(self, window_size: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.episode_successes = deque(maxlen=window_size)
        self.episode_crashes = deque(maxlen=window_size)
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check if an episode finished in any of the environments
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                
                # Classify success and crash based on standard reward signals
                is_success = ep_reward >= 200.0
                is_crash = ep_reward <= -100.0
                
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self.episode_successes.append(1.0 if is_success else 0.0)
                self.episode_crashes.append(1.0 if is_crash else 0.0)
                
                # Log metrics to TensorBoard
                if len(self.episode_rewards) > 0:
                    mean_reward = np.mean(self.episode_rewards)
                    mean_len = np.mean(self.episode_lengths)
                    success_rate = np.mean(self.episode_successes)
                    crash_rate = np.mean(self.episode_crashes)
                    
                    self.logger.record("rollout/ep_rew_mean_rolling", mean_reward)
                    self.logger.record("rollout/ep_len_mean_rolling", mean_len)
                    self.logger.record("rollout/success_rate_rolling", success_rate)
                    self.logger.record("rollout/crash_rate_rolling", crash_rate)
                    
                    if self.verbose > 0 and self.episode_count % 10 == 0:
                        print(f"[Callback] Ep {self.episode_count} finished | "
                              f"Rollout Reward Mean: {mean_reward:.2f} | "
                              f"Success Rate: {success_rate * 100:.1f}% | "
                              f"Crash Rate: {crash_rate * 100:.1f}%")
        return True
