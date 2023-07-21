"""
Offline reinforcement learning occurs when deployment and training
happen on different servers.
For example, the video game and the AI agent run on Windows,
but training must take place on a Linux server.
Usually, the deployed agent cannot access an encapsulated `env.step` function.

The agent must be stateful (keep track of observation/action state)

We use `gamma` for the MDP discount factor
and `gae_lambda` for the GAE discount factor.
"""

import gymnasium as gym

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.distributions import Categorical


class DummyAgent:
    def __init__(self, action_func, **kwargs):
        self.action_func = action_func

    def observe(self, obs):
        pass

    def act(self):
        # this function doesn't modify internal state
        return self.action_func()

    def learn(self, episodes):
        for i, (obs_list, act_list, reward_list) in enumerate(episodes):
            pass


def gae(rewards: np.ndarray, critic_values: np.ndarray, gamma: float, gae_lambda: float):
    n_steps = len(rewards)
    advantages = np.zeros(n_steps)
    advantages[-1] = rewards[-1] - critic_values[-1]
    for t in reversed(range(n_steps - 1)):  # from n_steps-2 to 0 inclusive
        delta = rewards[t] + gamma * critic_values[t + 1] - critic_values[t]
        advantages[t] = delta + gamma * gae_lambda * advantages[t + 1]
    return advantages


class VPG:
    # vanilla policy gradient
    def __init__(self, state_dim, action_dim, gamma=0.9, learning_rate=0.01, batch_size=32):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.policy_net = self.build_policy_net()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma

        self.current_obs = None

    def build_policy_net(self):
        policy_net = nn.Sequential(
            nn.Linear(self.state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim),
        )
        return policy_net

    def observe(self, obs):
        self.current_obs = torch.tensor(obs, dtype=torch.float32)

    def act(self):
        with torch.no_grad():
            # [1, n_actions]
            # action_probs = self.policy_net(self.current_obs).unsqueeze(0)
            # action = Categorical(action_probs).sample().item()
            action_logits = self.policy_net(self.current_obs).unsqueeze(0)
            action = Categorical(logits=action_logits).sample().item()
        return action

    def collate(self, episodes):
        """
        Collate a list of episodes into a single dataset.

        Parameters
        ----------
        episodes : list of tuple
            Each episode is a (observation_list, action_list, reward_list) tuple.

        Returns
        -------
        Dataset
            Can generate (observation, action, advantage) batches.
        """
        # Every observation has the same shape
        # (if not, extra processing is required)
        all_obs = []
        # Every action has the same shape
        all_act = []
        # Every reward (scalar) has the same shape,
        # but advantage is computed relative to one episode,
        # so it must be preprocessed before adding to the list
        all_adv = []
        for obs_list, act_list, reward_list in episodes:
            # in rare cases, the last state in truncated cases is
            n_steps = len(reward_list)
            all_obs.extend(obs_list[:n_steps])
            all_act.extend(act_list)
            # critic_values=0.0 with gae_lambda=1.0 is equivalent to Monte-carlo estimate
            all_adv.append(gae(
                rewards=np.array(reward_list), critic_values=np.zeros(n_steps),
                gamma=self.gamma, gae_lambda=1.0)
            )
        # start collating
        all_obs = np.array(all_obs)
        all_act = np.array(all_act)
        all_adv = np.hstack(all_adv)

        dataset = TensorDataset(
            torch.tensor(all_obs),
            torch.tensor(all_act),
            torch.tensor(all_adv)
        )
        return dataset

    def learn(self, episodes):
        dataset = self.collate(episodes)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for obs_batch, act_batch, reward_batch in dataloader:
            action_logits = self.policy_net(obs_batch)
            log_probs = Categorical(logits=action_logits).log_prob(act_batch)
            policy_loss = -(log_probs * reward_batch).mean()
            policy_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


def run_offline(env, agent, episodes_per_learn=1000, max_frames=100000):
    episode_count = 0
    last_n_rewards = []

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
            last_n_rewards.append(total_reward)
            n = min(30, len(last_n_rewards))
            avg = sum(last_n_rewards[-n:]) / n
            improvement_emoji = "🔥" if (total_reward > avg) else "😢"
            improvement_emoji = f"{episode_count:3} " + improvement_emoji
            print(
                f"{improvement_emoji} Finished with reward {int(total_reward)}.\tAverage of last {n}: {int(avg)}"
            )
            if len(episodes) >= episodes_per_learn:
                agent.learn(episodes)
                episodes = []

    from matplotlib import pyplot as plt
    plt.scatter(np.arange(len(last_n_rewards)) // episodes_per_learn * episodes_per_learn,
                last_n_rewards)
    plt.show()


def run_lunar_lander(visualize=True):
    env = gym.make("LunarLander-v2", render_mode="human" if visualize else None)

    agent = VPG(state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                gamma=0.95,
                learning_rate=0.003,
                batch_size=64)

    run_offline(env, agent, episodes_per_learn=5, max_frames=150_000)


def run_cart_pole(visualize=True):
    env = gym.make("CartPole-v1", render_mode="human" if visualize else None)
    # env = gym.make("MountainCar-v0", render_mode="human" if visualize else None)
    # env = gym.make("Acrobot-v1", render_mode="human" if visualize else None)
    # env = gym.make("Pendulum-v1", render_mode="human" if visualize else None)

    agent = VPG(state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                gamma=0.95,
                learning_rate=3e-3,
                batch_size=128)

    # agent = DummyAgent(action_func=lambda: np.random.randint(0, env.action_space.n))

    run_offline(env, agent, episodes_per_learn=5, max_frames=150_000)


if __name__ == '__main__':
    run_cart_pole(visualize=False)

