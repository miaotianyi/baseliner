"""
Offline reinforcement learning occurs when deployment and training
happen on different servers.
For example, the video game and the AI agent run on Windows,
but training must take place on a Linux server.
Usually, the deployed agent cannot access an encapsulated `env.step` function.

The agent must be stateful (keep track of observation/action state)

We use `gamma` for the MDP discount factor,
since `lambda` is reserved for lambda functions in Python.

"""

import gymnasium as gym

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
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


def gae_advantages(rewards: torch.Tensor, critic_values: torch.Tensor, gamma: float, gae_lambda: float):
    # Single trajectory generalized advantage estimation
    # Without gradients, compute a list of advantage based on
    # estimated values and rewards per timestep.
    # rewards, critic_values should have the same length n (the last terminal state is excluded)
    # it's better to calculate the entire thing in CPU
    with torch.no_grad():
        n_steps = len(rewards)
        old_device = rewards.device
        # rewards = torch.tensor(rewards, device="cpu")
        rewards = rewards.detach().clone().cpu()
        critic_values = critic_values.detach().clone().cpu()
        advantages = torch.zeros(n_steps, device="cpu")
        advantages[-1] = rewards[-1] - critic_values[-1]
        for t in reversed(range(n_steps-1)):    # from n_steps-2 to 0 inclusive
            delta = rewards[t] + gamma * critic_values[t + 1] - critic_values[t]
            advantages[t] = delta + gamma * gae_lambda * advantages[t + 1]
        return advantages.to(device=old_device)


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
            # nn.Softmax(dim=-1)
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

    def learn(self, episodes):
        for i, (obs_list, act_list, reward_list) in enumerate(episodes):
            obs_list = np.array(obs_list)
            act_list = np.array(act_list)
            reward_list = np.array(reward_list)
            obs_tensor = torch.tensor(obs_list, dtype=torch.float32)
            act_tensor = torch.tensor(act_list, dtype=torch.long)
            reward_tensor = torch.tensor(reward_list, dtype=torch.float32)

            adv_tensor = gae_advantages(rewards=reward_tensor, critic_values=torch.zeros_like(reward_tensor), gamma=self.gamma, gae_lambda=1.0)

            dataset = TensorDataset(obs_tensor, act_tensor, adv_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            for obs_batch, act_batch, reward_batch in dataloader:
                action_logits = self.policy_net(obs_batch)
                log_probs = Categorical(logits=action_logits).log_prob(act_batch)
                # action_probs = self.policy_net(obs_batch)  # [batch_size, 1]
                # log_probs = Categorical(action_probs).log_prob(act_batch)
                # log_probs = torch.log(torch.gather(action_probs, 1, act_batch.unsqueeze(1))).squeeze()
                # print((log_probs_other - log_probs).abs().mean())
                policy_loss = -(log_probs * reward_batch).mean()

                policy_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()


def run_offline(env, agent, episodes_per_learn=1000, max_frames=100000, ):
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
            total_reward = sum(reward_list)

            episodes.append((obs_list, act_list, reward_list))
            obs_list, act_list, reward_list = [], [], []
            obs, info = env.reset()
            last_n_rewards.append(total_reward)
            n = min(30, len(last_n_rewards))
            avg = sum(last_n_rewards[-n:]) / n
            improvement_emoji = "🔥" if (total_reward > avg) else "😢"
            print(
                f"{improvement_emoji} Finished with reward {int(total_reward)}.\tAverage of last {n}: {int(avg)}"
            )
            if len(episodes) >= episodes_per_learn:
                agent.learn(episodes)
                episodes = []


def run_lunar_lander(visualize=True):
    env = gym.make("LunarLander-v2", render_mode="human" if visualize else None)

    agent = VPG(state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                gamma=0.95,
                learning_rate=0.001,
                batch_size=32)

    run_offline(env, agent, episodes_per_learn=5, max_frames=50_000)


def run_cart_pole(visualize=True):
    env = gym.make("CartPole-v1", render_mode="human" if visualize else None)

    agent = VPG(state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                gamma=1.0,
                learning_rate=3e-3,
                batch_size=64)

    # agent = DummyAgent(action_func=lambda: np.random.randint(0, env.action_space.n))

    run_offline(env, agent, episodes_per_learn=5, max_frames=150_000)


if __name__ == '__main__':
    run_cart_pole(visualize=False)

