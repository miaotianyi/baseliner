"""
Utility functions that guarantee correctness

"""
import numpy as np


def gae(rewards: np.ndarray, critic_values: np.ndarray, gamma: float, gae_lambda: float):
    # Generalized advantage estimation
    # GAE[lambda](t) = ReturnsTD[lambda](t) - EstimatedValues(t)
    # The same gae_lambda is used for GAE and TD-lambda returns
    # Typical setting: gamma=0.99, gae_lambda=0.95
    n_steps = len(rewards)
    advantages = np.zeros(n_steps)
    advantages[-1] = rewards[-1] - critic_values[n_steps-1]
    # if len(critic_values) == n_steps + 1:   # truncated
    #     advantages[-1] += gamma * critic_values[n_steps]
    for t in reversed(range(n_steps - 1)):  # from n_steps-2 to 0 inclusive
        delta = rewards[t] + gamma * critic_values[t + 1] - critic_values[t]
        advantages[t] = delta + gamma * gae_lambda * advantages[t + 1]
    return advantages
