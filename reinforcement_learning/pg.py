"""
Simplest policy gradient (Reinforce algorithm)
"""
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from reinforcement_learning.run_offline import run_offline
from reinforcement_learning.utils import gae


class DummyAgent:
    def __init__(self, action_func, **kwargs):
        self.action_func = action_func

    def observe(self, obs):
        pass

    def act(self):
        # this function doesn't modify internal state
        return self.action_func()

    def learn(self, episodes):
        # Usually, there's also a self.collate method
        # that converts a list of episodes to a PyTorch Dataset
        for i, (obs_list, act_list, reward_list) in enumerate(episodes):
            pass


class CatMLP(nn.Module):
    def __init__(self, n_features, n_actions, d=64, lr=1e-3):
        super(CatMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, n_actions)
        )
        self.lr = lr

    def sample(self, obs):
        logits = self.net(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()

    def score(self, obs, action):
        logits = self.net(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action_log_prob = dist.log_prob(action.flatten())
        return action_log_prob

    def make_optimizer(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-5)
        return optimizer


class VPG:
    # vanilla policy gradient
    def __init__(self,
                 policy: nn.Module,
                 gamma: float,
                 batch_size: int):
        self.policy = policy
        self.optimizer = self.policy.make_optimizer()

        self.batch_size = batch_size
        self.gamma = gamma
        self.current_obs = None

    def observe(self, obs):
        self.current_obs = np.array(obs)

    def act(self):
        with torch.no_grad():
            # [1, n_actions]
            # action_probs = self.policy_net(self.current_obs).unsqueeze(0)
            # action = Categorical(action_probs).sample().item()
            obs = torch.tensor(self.current_obs, dtype=torch.float32)
            action = self.policy.sample(obs).item()
            # logits = self.policy(obs).unsqueeze(0)
            # action = torch.distributions.Categorical(logits=logits).sample().item()
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
            log_probs = self.policy.score(obs_batch, act_batch)
            # logits = self.policy(obs_batch)
            # log_probs = torch.distributions.Categorical(logits=logits).log_prob(act_batch)
            policy_loss = -(log_probs * reward_batch).mean()
            policy_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


def run_cart_pole(visualize=True):
    import gymnasium as gym
    env = gym.make("CartPole-v1", render_mode="human" if visualize else None)
    # env = gym.make("MountainCar-v0", render_mode="human" if visualize else None)
    # env = gym.make("Acrobot-v1", render_mode="human" if visualize else None)
    # env = gym.make("Pendulum-v1", render_mode="human" if visualize else None)
    policy = CatMLP(
        n_features=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        d=32,
        lr=1e-3
    )
    agent = VPG(
        policy=policy,
        gamma=0.95,
        batch_size=128
    )
    run_offline(env, agent, episodes_per_learn=5, max_frames=150_000)


if __name__ == '__main__':
    run_cart_pole(visualize=False)
