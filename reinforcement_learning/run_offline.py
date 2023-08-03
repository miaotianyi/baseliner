"""
Distributed reinforcement learning occurs when deployment and training
happen on different servers.
For example, the video game and the AI agent run on Windows,
but training must take place on a Linux server.
Usually, the deployed agent cannot access an encapsulated `env.step` function.

The agent must be stateful (keep track of observation/action state)

We use `gamma` for the MDP discount factor
and `gae_lambda` for the GAE discount factor.

This environment
"""

import numpy as np


def run_offline(env, agent, episodes_per_learn=1000, max_frames=100000):
    episode_count = 0
    episode_returns = []

    episodes = []
    obs_list, act_list, reward_list = [], [], []
    obs, info = env.reset()
    for _ in range(max_frames):
        agent.observe(obs)
        action = agent.act()
        next_obs, reward, terminated, truncated, info = env.step(action)
        obs_list.append(obs)
        act_list.append(action)
        reward_list.append(reward)
        obs = next_obs
        if terminated or truncated:
            # Last-value bootstrapping actually hurts performance in CartPole,
            # so it's disabled by default.
            # if truncated:
            #     obs_list.append(obs)
            total_reward = sum(reward_list)
            episode_count += 1

            episodes.append((obs_list, act_list, reward_list))
            obs_list, act_list, reward_list = [], [], []
            obs, info = env.reset()
            episode_returns.append(total_reward)

            n = min(30, len(episode_returns))
            avg = sum(episode_returns[-n:]) / n

            improvement_emoji = "🔥" if (total_reward > avg) else "😢"
            improvement_emoji = f"{episode_count:3} " + improvement_emoji
            print(
                f"{improvement_emoji} Finished with reward {int(total_reward)}.\tAverage of last {n}: {int(avg)}"
            )
            if len(episodes) >= episodes_per_learn:
                agent.learn(episodes)
                episodes = []

    from matplotlib import pyplot as plt
    plt.scatter(np.arange(len(episode_returns)) // episodes_per_learn * episodes_per_learn,
                episode_returns,
                alpha=0.5)
    # plot average returns for each learning cycle
    avg_returns = [np.mean(episode_returns[i:i + episodes_per_learn])
                   for i in range(0, len(episode_returns), episodes_per_learn)]
    plt.plot(np.arange(len(avg_returns)) * episodes_per_learn,
             avg_returns)
    plt.xlabel("Episode")
    plt.ylabel("Sum of rewards")
    plt.show()


