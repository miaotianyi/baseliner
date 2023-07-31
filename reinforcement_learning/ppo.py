"""
We use "policy" to denote the neural network architecture (nn.Module)
and "agent" to denote the training loop boilerplate.
"""
import copy

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from reinforcement_learning.utils import gae


def clipped_ppo_loss(actor_log_probs, old_log_probs, old_advantages, clip):
    """

    Parameters
    ----------
    actor_log_probs: torch.Tensor
        1D tensor of log probability of taking the action at each time step;
        tracks gradient.

    old_log_probs: torch.Tensor
        Action log probability scored by the old network.

    old_advantages: torch.Tensor
        Action advantages scored by the old network

    clip: float

    Returns
    -------
    scalar
    """
    ratio = torch.exp(actor_log_probs - old_log_probs)
    clipped_ratio = ratio.clamp(min=1.0 - clip, max=1.0 + clip)
    # objective: bigger is better!
    policy_objective = torch.minimum(
        ratio * old_advantages, clipped_ratio * old_advantages
    )
    return -policy_objective.mean()


def clipped_vf_loss(
    critic_values: torch.Tensor,
    old_values: torch.Tensor,
    old_returns: torch.Tensor,
    clip: float,
):
    """
    Clipped value function loss.

    central idea: old advantages/old values are based on the current (old) network
    it's only computed once for all current trajectories in this agent-epoch
    the old network is copied from the new network at the end of each agent-epoch
    The critic values change per PPO-epoch.

    In an agent-epoch, the agent interacts with the env to obtain trajectories,
    then updates its internal parameters.
    In a PPO-epoch, the parameters of the network are updated once,
    by iterating over mini-batches of current trajectories


    Parameters
    ----------
    critic_values: torch.Tensor
        The only input argument with gradients tracked.

        Has length n, where n is the number of time steps in the trajectory.
        (not including the terminal state)

    old_values: torch.Tensor
        The value estimation by old critic.
        Has length n, where n is the number of time steps in the trajectory.

    old_returns: torch.Tensor
        The returns calculated by the old critic (value function)
        from previous agent-epoch.

        Precomputed as `old_returns = old_values + old_advantages`.

        Has length n, where n is the number of time steps in the trajectory.

    clip: float
        The positive clip hyperparameter

    Returns
    -------
    scalar
    """
    # cannot deviate too much from old_values
    clipped_values = old_values + (critic_values - old_values).clamp(min=-clip, max=clip)
    vf_loss = torch.maximum(
        (critic_values - old_returns) ** 2, (clipped_values - old_returns) ** 2
    )
    return 0.5 * vf_loss.mean()


class CatPolicyMLP(nn.Module):
    def __init__(self, n_features, n_actions, d=64,
                 actor_lr=2.5e-4, critic_lr=1e-3, default_lr=1e-3):
        """
        A simple MLP backbone for a PPO agent,
        where the input features are all numerical,
        and the output is the choice within n actions.
        """
        super().__init__()
        self.d = d
        self.n_features = n_features
        self.n_actions = n_actions
        self.extractor = nn.Sequential(
            nn.Linear(n_features, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            # nn.Linear(d, d),
            # nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.default_lr = default_lr

    def sample(self, obs):
        obs = self.extractor(obs)
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().item()

    def score(self, obs, action):
        # intermediate representation
        x = self.extractor(obs)
        # critic value for state/observation
        estimated_value = self.critic(x).flatten()
        # action logits
        logits = self.actor(x)
        # action distribution
        dist = torch.distributions.Categorical(logits=logits)
        # action log probability
        action_log_prob = dist.log_prob(action.flatten())
        # action entropy
        entropy = dist.entropy()
        return estimated_value, action_log_prob, entropy

    def make_optimizer(self):
        optimizer = optim.Adam([
            {"params": self.extractor.parameters()},
            {"params": self.actor.parameters(), "lr": self.actor_lr},
            {"params": self.critic.parameters(), "lr": self.critic_lr}],
            lr=self.default_lr, eps=1e-5)
        return optimizer


class Func(nn.Module):
    def __init__(self, func):
        super(Func, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class GaussianMLP(nn.Module):
    def __init__(self, n_features, n_actions, d=64,
                 low=None, high=None,
                 actor_lr=2.5e-4, critic_lr=1e-3, default_lr=1e-3):
        """
        A simple MLP backbone for a PPO agent,
        where the input features are all numerical,
        and the output is a continuous action space.
        """
        super().__init__()
        self.low = np.array(low)
        self.high = np.array(high)
        self.d = d
        self.n_features = n_features
        self.n_actions = n_actions
        self.extractor = nn.Sequential(
            nn.Linear(n_features, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, n_actions),
            Func(lambda x: 2 * torch.tanh(x))
        )
        self.actor_log_std = nn.Parameter(-0.5 * torch.ones(n_actions))
        self.critic = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.default_lr = default_lr

    def sample(self, obs):
        x = self.extractor(obs)
        mean = self.actor_mean(x)
        std = self.actor_log_std.exp()
        dist = torch.distributions.Normal(loc=mean, scale=std)
        action = dist.sample()
        action = action.cpu().numpy()
        action = np.clip(action, a_min=self.low, a_max=self.high)
        return action

    def score(self, obs, action):
        # intermediate representation
        x = self.extractor(obs)
        # critic value for state/observation
        estimated_value = self.critic(x).flatten()
        # action mean
        mean = self.actor_mean(x)
        # action std
        std = self.actor_log_std.exp()
        # action distribution
        dist = torch.distributions.Normal(loc=mean, scale=std)
        # action log probability
        action_log_prob = dist.log_prob(action).flatten()
        # action entropy
        entropy = dist.entropy()
        return estimated_value, action_log_prob, entropy

    def make_optimizer(self):
        optimizer = optim.Adam([
            {"params": self.extractor.parameters()},
            {"params": self.actor_mean.parameters(), "lr": self.actor_lr},
            {"params": self.actor_log_std, "lr": self.actor_lr},
            {"params": self.critic.parameters(), "lr": self.critic_lr}],
            lr=self.default_lr, eps=1e-5)
        return optimizer


class PPO:
    def __init__(self,
                 policy: nn.Module,
                 gamma: float,
                 gae_lambda: float,
                 ppo_epochs: int,
                 batch_size: int,
                 vf_weight: float,
                 entropy_weight: float,
                 ppo_clip: float,
                 vf_clip: float,
                 ):
        # Initialize policy network
        self.policy = policy
        # Initialize optimizer
        self.optimizer = self.policy.make_optimizer()
        # old policy is always in no-grad and eval mode.
        self.old_policy = copy.deepcopy(policy)
        # No-grad and eval() will persist through
        # `self.old_policy.load_state_dict(self.policy.state_dict())`
        self.old_policy.eval()
        for p in self.old_policy.parameters():
            p.requires_grad_(False)

        # Save hyperparameters
        # MDP/RL returns discount
        self.gamma = gamma
        # TD(lambda) and GAE(lambda) discount
        self.gae_lambda = gae_lambda
        # Number of importance-sampled off-policy learning epochs
        self.ppo_epochs = ppo_epochs
        # Dataloader batch size
        self.batch_size = batch_size
        # (The weight for PPO loss is 1)
        # Weight of value function loss (always positive)
        self.vf_weight = vf_weight
        # Weight of entropy term (always positive)
        self.entropy_weight = entropy_weight
        # Clip coefficient for PPO ratio (0.0 to 1.0)
        self.ppo_clip = ppo_clip
        # Clip coefficient for value function loss
        self.vf_clip = vf_clip

        # Stateful agent: store current observation
        self.current_obs = None

    def observe(self, obs):
        self.current_obs = torch.tensor(obs, dtype=torch.float32)

    def act(self):
        with torch.no_grad():
            action = self.old_policy.sample(self.current_obs)
        return action

    def collate(self, episodes):
        # observations, actions, old values, old advantages, old returns, old log prob
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
        # Estimated values from old_policy
        all_val = []
        # Every reward (scalar) has the same shape,
        # but advantage is computed relative to one episode,
        # so it must be preprocessed before adding to the list
        all_adv = []
        # Sample returns is the sum of value and advantage
        all_ret = []
        # Sample log probability of taking those actions
        all_lp = []
        for obs_list, act_list, reward_list in episodes:
            # in rare cases, the last state in truncated cases is
            n_steps = len(reward_list)
            all_obs.extend(obs_list[:n_steps])
            all_act.extend(act_list)

            obs_tensor = torch.tensor(np.array(obs_list[:n_steps]), dtype=torch.float32)
            act_tensor = torch.tensor(np.array(act_list[:n_steps]))
            with torch.no_grad():
                # ignore entropy
                val_tensor, lp_tensor, _ = self.old_policy.score(obs_tensor, act_tensor)
            val_array = val_tensor.cpu().numpy()
            lp_array = lp_tensor.cpu().numpy()
            adv_array = gae(rewards=np.array(reward_list), critic_values=val_array,
                            gamma=self.gamma, gae_lambda=self.gae_lambda)
            ret_array = val_array + adv_array

            all_val.append(val_array)
            all_adv.append(adv_array)
            all_ret.append(ret_array)
            all_lp.append(lp_array)

        # start collating
        all_obs = np.array(all_obs)
        all_act = np.array(all_act)
        all_val = np.hstack(all_val)
        all_adv = np.hstack(all_adv)
        all_ret = np.hstack(all_ret)
        all_lp = np.hstack(all_lp)

        dataset = TensorDataset(*[torch.tensor(x) for x in [
            all_obs, all_act, all_val, all_adv, all_ret, all_lp]])
        return dataset

    def learn(self, episodes):
        dataset = self.collate(episodes)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.ppo_epochs):
            for obs, act, val, adv, ret, lp in dataloader:
                new_val, new_lp, new_ent = self.policy.score(obs, act)
                ppo_loss = clipped_ppo_loss(
                    actor_log_probs=new_lp, old_log_probs=lp, old_advantages=adv, clip=self.ppo_clip)
                vf_loss = clipped_vf_loss(
                    critic_values=new_val, old_values=val, old_returns=ret, clip=self.vf_clip)
                entropy_loss = -new_ent.mean()  # bigger entropy, more explore, better
                loss = ppo_loss + self.vf_weight * vf_loss + self.entropy_weight * entropy_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.old_policy.load_state_dict(self.policy.state_dict())


def run_cart_pole():
    from reinforcement_learning.run_offline import run_offline
    import gymnasium as gym

    visualize = False

    env = gym.make("CartPole-v1", render_mode="human" if visualize else None)
    # env = gym.make("MountainCar-v0", render_mode="human" if visualize else None)
    # env = gym.make("Acrobot-v1", render_mode="human" if visualize else None)
    # env = gym.make("Pendulum-v1", render_mode="human" if visualize else None)
    # env = gym.make("LunarLander-v2", render_mode="human" if visualize else None)

    policy = CatPolicyMLP(
        n_features=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        d=32,
        actor_lr=1e-3,
        critic_lr=1e-3,
        default_lr=1e-3)
    agent = PPO(
        policy=policy,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=3,
        batch_size=64,
        vf_weight=0.5,
        entropy_weight=0.01,
        ppo_clip=0.2,
        vf_clip=3.0
    )

    run_offline(env, agent, episodes_per_learn=5, max_frames=150_000)


def run_pendulum():
    from reinforcement_learning.run_offline import run_offline
    import gymnasium as gym

    visualize = False

    # env = gym.make("Acrobot-v1", render_mode="human" if visualize else None)
    env = gym.make("Pendulum-v1", render_mode="human" if visualize else None)

    policy = GaussianMLP(
        n_features=env.observation_space.shape[0],
        n_actions=len(env.action_space.shape),
        d=64,
        low=-2,
        high=2,
        actor_lr=1e-3,
        critic_lr=1e-3,
        default_lr=1e-3)

    agent = PPO(
        policy=policy,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=3,
        batch_size=64,
        vf_weight=0.5,
        entropy_weight=0.01,
        ppo_clip=0.2,
        vf_clip=3.0
    )

    run_offline(env, agent, episodes_per_learn=5, max_frames=100_000)


if __name__ == '__main__':
    run_cart_pole()
