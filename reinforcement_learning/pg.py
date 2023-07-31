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
        return dist.sample().item()

    def score(self, obs, action):
        logits = self.net(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action_log_prob = dist.log_prob(action.flatten())
        return action_log_prob

    def make_optimizer(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-5)
        return optimizer


class CELU1p(nn.Module):
    def __init__(self, alpha=1.0, eps=1e-5):
        # To model a strictly positive quantity (e.g. standard deviation std in VAE),
        # people often let NN output log(std)
        # and calculate std = net(x).exp()
        # But this could result in overflow when net(x) gets bigger.
        super(CELU1p, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self.added = eps + 1.0

    def forward(self, x):
        return torch.celu(x, alpha=self.alpha) + self.added


class BetaMLP(nn.Module):
    def __init__(self, n_features, n_actions, d=64, lr=1e-3,
                 low=0.0, high=1.0):
        super(BetaMLP, self).__init__()
        # 4-argument Beta distribution
        # Useful for distributions bounded on 2 sides
        self.low = low
        self.range = high - low
        self.lr = lr

        self.net = nn.Sequential(
            nn.Linear(n_features, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, n_actions * 2),
            CELU1p()
        )

    def sample(self, obs):
        alpha, beta = torch.chunk(self.net(obs), chunks=2, dim=-1)
        dist = torch.distributions.Beta(concentration1=alpha, concentration0=beta)
        action = dist.sample().cpu().numpy()
        action = action * self.range + self.low
        return action

    def score(self, obs, action):
        action = (action - self.low) / self.range   # squeeze back to [0, 1]
        alpha, beta = torch.chunk(self.net(obs), chunks=2, dim=-1)
        dist = torch.distributions.Beta(concentration1=alpha, concentration0=beta)
        action_log_prob = dist.log_prob(action).flatten()
        return action_log_prob

    def make_optimizer(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-5)
        return optimizer


class GaussianMLP(nn.Module):
    def __init__(self, n_features, n_actions, d=64, lr=1e-3,
                 low=None, high=None):
        super(GaussianMLP, self).__init__()
        self.mean = nn.Sequential(
            nn.Linear(n_features, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, n_actions)
        )
        self.log_std = nn.Parameter(-0.5 * torch.ones(n_actions))
        self.lr = lr
        # lower and upper bound for continuous actions
        self.low = low
        self.high = high

    def sample(self, obs):
        mean = self.mean(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(loc=mean, scale=std)
        action = dist.sample().cpu().numpy()
        action = np.clip(action, a_min=self.low, a_max=self.high)
        return action

    def score(self, obs, action):
        mean = self.mean(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(loc=mean, scale=std)
        action_log_prob = dist.log_prob(action).flatten()
        return action_log_prob

    def make_optimizer(self):
        optimizer = optim.Adam(self.mean.parameters(), lr=self.lr, eps=1e-5)
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
            obs = torch.tensor(self.current_obs, dtype=torch.float32)
            action = self.policy.sample(obs)
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


def run_pendulum(visualize=True):
    import gymnasium as gym
    # env = gym.make("Acrobot-v1", render_mode="human" if visualize else None)
    env = gym.make("Pendulum-v1", render_mode="human" if visualize else None)
    policy = BetaMLP(
        n_features=env.observation_space.shape[0],
        n_actions=len(env.action_space.shape),
        d=64,
        lr=1e-4,
        low=-2.,
        high=2.,
    )
    agent = VPG(
        policy=policy,
        gamma=0.95,
        batch_size=128
    )
    run_offline(env, agent, episodes_per_learn=10, max_frames=100_000)


if __name__ == '__main__':
    run_pendulum(visualize=False)
