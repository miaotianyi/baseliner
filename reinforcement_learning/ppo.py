"""
We use "policy" to denote the neural network architecture (nn.Module)
and "agent" to denote the training loop boilerplate.
"""
import copy

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
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
    # classic policy objective (from PPO paper)
    # clipped_ratio = ratio.clamp(min=1.0 - clip, max=1.0 + clip)
    # policy_objective = torch.minimum(ratio * old_advantages, clipped_ratio * old_advantages)
    # simplified policy objective (from Spinning Up)
    g = torch.where(old_advantages >= 0, 1 + clip, 1 - clip) * old_advantages
    policy_objective = torch.minimum(ratio * old_advantages, g)

    # objective: bigger is better!
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
    # if clip is None:
    #     return 0.5 * ((critic_values - old_returns) ** 2).mean()
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


def mlp(dims, activation, output_activation=None):
    """
    Easily initialize an MLP.

    Copied from Spinning Up.
    """
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
        # act = activation if i < len(dims) - 2 else output_activation
        # layers += [nn.Linear(dims[i], dims[i + 1]), act()]
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)

class SepCatMLP(nn.Module):
    def __init__(self, n_features, n_actions,
                 actor_lr, critic_lr,
                 d=64):
        super().__init__()
        # separate actor and critic networks
        self.actor = mlp([n_features] + [d] * 2 + [n_actions], activation=nn.ReLU)
        self.critic = mlp([n_features] + [d] * 2 + [1], activation=nn.ReLU)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

    def sample(self, obs):
        # real-valued outputs (to be converted to strictly positive)
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
        return action

    def score(self, obs, action):
        # intermediate representation
        # critic value for state/observation
        estimated_value = self.critic(obs).flatten()

        # action distribution
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)

        # action log probability
        action_log_prob = dist.log_prob(action)
        # action entropy
        entropy = dist.entropy()
        return estimated_value, action_log_prob, entropy

    def make_optimizer(self):
        optimizer = optim.Adam([
            {"params": self.actor.parameters(), "lr": self.actor_lr},
            {"params": self.critic.parameters(), "lr": self.critic_lr}],
            eps=1e-5)
        return optimizer


# class BetaMLP(nn.Module):
#     def __init__(self, n_features, n_actions,
#                  low=0.0, high=1.0,
#                  d=64,
#                  actor_lr=2.5e-4, critic_lr=1e-3
#                  ):
#         super().__init__()
#         self.actor = mlp([n_features] + [d] * 2 + [n_actions * 2], activation=nn.ReLU)
#         self.critic = mlp([n_features] + [d] * 2 + [1], activation=nn.ReLU)
#
#         self.low = low
#         self.range = high - low
#         self.actor_lr = actor_lr
#         self.critic_lr = critic_lr
#
#     @staticmethod
#     def real_to_positive(reals):
#         # softplus result: around 250 running mean
#         # exp result: around 170 running mean
#         # softplus seems superior for lunar lander
#         # Softplus works better for continuous lunar lander,
#         # but worse for pendulum (comparing with exp[-200])
#         # return F.softplus(reals) + 1.0
#         # return F.elu(reals) + 2.0
#         return torch.exp(reals) + 1.0
#
#     def sample(self, obs):
#         # real-valued outputs (to be converted to strictly positive)
#         reals = self.actor(obs)
#         positive = self.real_to_positive(reals)
#         alpha, beta = torch.chunk(positive, chunks=2, dim=-1)
#         dist = torch.distributions.Beta(concentration1=alpha, concentration0=beta)
#         action = dist.sample()
#         action = action.cpu().numpy()
#         action = action * self.range + self.low
#         return action
#
#     def score(self, obs, action):
#         # intermediate representation
#         # critic value for state/observation
#         estimated_value = self.critic(obs).flatten()
#
#         # action distribution
#         reals = self.actor(obs)
#         positive = self.real_to_positive(reals)
#         alpha, beta = torch.chunk(positive, chunks=2, dim=-1)
#         dist = torch.distributions.Beta(concentration1=alpha, concentration0=beta)
#
#         # action log probability
#         action = (action - self.low) / self.range   # squeeze back to [0, 1]
#         action_log_prob = dist.log_prob(action).sum(dim=-1)
#         # action entropy
#         entropy = dist.entropy()
#         return estimated_value, action_log_prob, entropy
#
#     def make_optimizer(self):
#         optimizer = optim.Adam([
#             {"params": self.actor.parameters(), "lr": self.actor_lr},
#             {"params": self.critic.parameters(), "lr": self.critic_lr}],
#             eps=1e-5)
#         return optimizer


class SepBetaMLP(nn.Module):
    def __init__(self, n_features, n_actions,
                 low, high,
                 net_arch=(64, 64),
                 actor_lr=2.5e-4, critic_lr=1e-3
                 ):
        super().__init__()
        self.actor_alpha = mlp([n_features] + list(net_arch) + [n_actions], activation=nn.ReLU)
        self.actor_beta = mlp([n_features] + list(net_arch) + [n_actions], activation=nn.ReLU)
        self.critic = mlp([n_features] + list(net_arch) + [1], activation=nn.ReLU)

        self.low = low
        self.range = high - low
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

    @staticmethod
    def real_to_positive(reals):
        # softplus result: around 250 running mean
        # exp result: around 170 running mean
        # softplus seems superior for lunar lander
        # Softplus works better for continuous lunar lander,
        # but worse for pendulum (comparing with exp[-200])
        return F.softplus(reals) + 1.0
        # return F.elu(reals) + 2.0
        # return torch.exp(reals) + 1.0

    def sample(self, obs):
        # real-valued outputs (to be converted to strictly positive)
        alpha = self.real_to_positive(self.actor_alpha(obs))
        beta = self.real_to_positive(self.actor_beta(obs))
        dist = torch.distributions.Beta(concentration1=alpha, concentration0=beta)
        action = dist.sample()
        action = action.cpu().numpy()
        action = action * self.range + self.low
        return action

    def score(self, obs, action):
        # intermediate representation
        # critic value for state/observation
        estimated_value = self.critic(obs).flatten()

        # action distribution
        alpha = self.real_to_positive(self.actor_alpha(obs))
        beta = self.real_to_positive(self.actor_beta(obs))
        dist = torch.distributions.Beta(concentration1=alpha, concentration0=beta)

        # action log probability
        action = (action - self.low) / self.range   # squeeze back to [0, 1]
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        # action entropy
        entropy = dist.entropy()
        return estimated_value, action_log_prob, entropy

    def make_optimizer(self):
        optimizer = optim.Adam([
            {"params": self.actor_alpha.parameters(), "lr": self.actor_lr},
            {"params": self.actor_beta.parameters(), "lr": self.actor_lr},
            {"params": self.critic.parameters(), "lr": self.critic_lr}],
            eps=1e-5)
        return optimizer


class SqueezeMLP(nn.Module):
    def __init__(self, n_features, n_actions,
                 low, high,
                 net_arch=(64, 64),
                 actor_lr=2.5e-4, critic_lr=1e-3):
        super().__init__()
        self.actor_mean = mlp(
            [n_features] + list(net_arch) + [n_actions],
            activation=nn.ReLU, output_activation=nn.Sigmoid)
        # self.actor_log_std = mlp(
        #     [n_features] + list(net_arch) + [n_actions],
        #     activation=nn.ReLU)
        self.actor_log_std = nn.Parameter(-torch.ones(n_actions))
        self.critic = mlp([n_features] + list(net_arch) + [1], activation=nn.ReLU)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.low = low
        self.range = high - low

    def sample(self, obs):
        # x = self.extractor(obs)
        mean = self.actor_mean(obs)
        # std = self.actor_log_std(obs).exp()
        std = self.actor_log_std.exp()
        dist = torch.distributions.Normal(loc=mean, scale=std)
        action = dist.sample()
        action = action.cpu().numpy()
        action = action * self.range + self.low     # special for squeezed
        return action

    def score(self, obs, action):
        # intermediate representation
        # x = self.extractor(obs)
        # critic value for state/observation
        estimated_value = self.critic(obs).flatten()
        # action mean
        mean = self.actor_mean(obs)
        # action std
        # std = self.actor_log_std(obs).exp()
        std = self.actor_log_std.exp()
        # action distribution
        dist = torch.distributions.Normal(loc=mean, scale=std)
        # action log probability
        action = (action - self.low) / self.range   # special for squeezed
        action_log_prob = dist.log_prob(action).sum(dim=-1)

        # action entropy
        entropy = dist.entropy()
        return estimated_value, action_log_prob, entropy

    def make_optimizer(self):
        optimizer = optim.Adam([
            {"params": self.actor_mean.parameters(), "lr": self.actor_lr},
            # {"params": self.actor_log_std.parameters(), "lr": self.actor_lr},
            {"params": self.actor_log_std, "lr": self.actor_lr},
            {"params": self.critic.parameters(), "lr": self.critic_lr}],
            eps=1e-5)
        return optimizer


class GaussianMLP(nn.Module):
    def __init__(self, n_features, n_actions,
                 net_arch=(64, 64),
                 actor_lr=2.5e-4, critic_lr=1e-3):
        """
        A simple MLP backbone for a PPO agent,
        where the input features are all numerical,
        and the output is a continuous action space.
        """
        super().__init__()
        self.actor_mean = mlp([n_features] + list(net_arch) + [n_actions], activation=nn.ReLU)
        self.actor_log_std = mlp([n_features] + list(net_arch) + [n_actions], activation=nn.ReLU)
        self.critic = mlp([n_features] + list(net_arch) + [1], activation=nn.ReLU)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

    def sample(self, obs):
        # x = self.extractor(obs)
        mean = self.actor_mean(obs)
        std = self.actor_log_std(obs).exp()
        dist = torch.distributions.Normal(loc=mean, scale=std)
        action = dist.sample()
        action = action.cpu().numpy()
        # action = np.clip(action, a_min=self.low, a_max=self.high)
        return action

    def score(self, obs, action):
        # intermediate representation
        # x = self.extractor(obs)
        # critic value for state/observation
        estimated_value = self.critic(obs).flatten()
        # action mean
        mean = self.actor_mean(obs)
        # action std
        std = self.actor_log_std(obs).exp()
        # action distribution
        dist = torch.distributions.Normal(loc=mean, scale=std)
        # action log probability
        action_log_prob = dist.log_prob(action).sum(dim=-1)

        # action entropy
        entropy = dist.entropy()
        return estimated_value, action_log_prob, entropy

    def make_optimizer(self):
        optimizer = optim.Adam([
            # {"params": self.extractor.parameters()},
            {"params": self.actor_mean.parameters(), "lr": self.actor_lr},
            {"params": self.actor_log_std.parameters(), "lr": self.actor_lr},
            {"params": self.critic.parameters(), "lr": self.critic_lr}],
            eps=1e-5)
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

        # debug tool:
        # print([x.shape for x in [
        #     all_obs, all_act, all_val, all_adv, all_ret, all_lp]])

        dataset = TensorDataset(*[torch.tensor(x) for x in [
            all_obs, all_act, all_val, all_adv, all_ret, all_lp]])
        return dataset

    def learn(self, episodes):
        dataset = self.collate(episodes)
        # drop_last prevents "NaN std" during advantage normalization
        # Not much information is lost because shuffle is random
        # and there are multiple PPO epochs.
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            drop_last=True
        )
        for _ in range(self.ppo_epochs):
            for obs, act, val, adv, ret, lp in dataloader:
                new_val, new_lp, new_ent = self.policy.score(obs, act)
                adv = (adv - adv.mean()) / adv.std()
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


def run_cart_pole(visualize=False):
    from reinforcement_learning.run_offline import run_offline
    import gymnasium as gym

    env = gym.make("CartPole-v1", render_mode="human" if visualize else None)

    policy = SepCatMLP(
        n_features=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        d=32,
        actor_lr=1e-3,
        critic_lr=1e-3
    )

    agent = PPO(
        policy=policy,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=3,
        batch_size=128,
        vf_weight=0.5,
        entropy_weight=0.01,
        ppo_clip=0.2,
        vf_clip=1000.0
    )

    run_offline(env, agent, episodes_per_learn=10, max_frames=150_000)


def run_lunar_lander(visualize=False):
    from reinforcement_learning.run_offline import run_offline
    import gymnasium as gym

    env = gym.make("LunarLander-v2", render_mode="human" if visualize else None)

    policy = SepCatMLP(
        n_features=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        d=128,
        actor_lr=1e-3,
        critic_lr=1e-3
    )
    agent = PPO(
        policy=policy,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=3,
        batch_size=128,
        vf_weight=0.5,
        entropy_weight=0.01,
        ppo_clip=0.2,
        vf_clip=10.0
    )
    run_offline(env, agent, episodes_per_learn=10, max_frames=1_000_000)


def run_pendulum(visualize=False):
    from reinforcement_learning.run_offline import run_offline
    import gymnasium as gym

    env = gym.make("Pendulum-v1", render_mode="human" if visualize else None)

    # policy = SepBetaMLP(
    #     n_features=env.observation_space.shape[0],
    #     n_actions=1,
    #     low=-2.0,
    #     high=2.0,
    #     net_arch=[128, 128],
    #     actor_lr=3e-4,
    #     critic_lr=3e-4
    # )
    policy = GaussianMLP(
        n_features=env.observation_space.shape[0],
        n_actions=1,
        net_arch=[128, 128],
        actor_lr=3e-4,
        critic_lr=3e-4
    )

    agent = PPO(
        policy=policy,
        gamma=0.9,
        gae_lambda=0.95,
        ppo_epochs=3,
        batch_size=128,
        vf_weight=0.5,
        entropy_weight=0.01,
        ppo_clip=0.2,
        vf_clip=100.0
    )

    run_offline(env, agent, episodes_per_learn=10, max_frames=500_000)


def run_lunar_lander_continuous(visualize=False):
    from reinforcement_learning.run_offline import run_offline
    import gymnasium as gym

    env = gym.make("LunarLander-v2", continuous=True)

    policy = SepBetaMLP(
        n_features=env.observation_space.shape[0],
        n_actions=2,
        low=-1.,
        high=1.,
        net_arch=[128]*2,
        actor_lr=1e-3,
        critic_lr=1e-3
    )
    # policy = GaussianMLP(
    #     n_features=env.observation_space.shape[0],
    #     n_actions=2,
    #     net_arch=[128]*2,
    #     actor_lr=1e-3,
    #     critic_lr=1e-3
    # )

    agent = PPO(
        policy=policy,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=3,
        batch_size=128,
        vf_weight=0.5,
        entropy_weight=0.01,
        ppo_clip=0.2,
        vf_clip=10.0
    )

    run_offline(env, agent, episodes_per_learn=10, max_frames=1_000_000)


def run_mountain_car_continuous(visualize=False):
    from reinforcement_learning.run_offline import run_offline
    import gymnasium as gym

    env = gym.make("MountainCarContinuous-v0", render_mode="human" if visualize else None)
    env._max_episode_steps = 10000

    policy = GaussianMLP(
        n_features=env.observation_space.shape[0],
        n_actions=1,
        net_arch=[128]*2,
        actor_lr=1e-3,
        critic_lr=1e-3
    )

    agent = PPO(
        policy=policy,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=3,
        batch_size=64,
        vf_weight=0.5,
        entropy_weight=0.01,
        ppo_clip=0.2,
        vf_clip=1000.0
    )

    run_offline(env, agent, episodes_per_learn=10, max_frames=500_000)


def run_acrobot(visualize=False):
    from reinforcement_learning.run_offline import run_offline
    import gymnasium as gym

    env = gym.make("Acrobot-v1", render_mode="human" if visualize else None)

    policy = SepCatMLP(
        n_features=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        actor_lr=1e-3,
        critic_lr=1e-3,
        d=256
    )

    agent = PPO(
        policy=policy,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=3,
        batch_size=64,
        vf_weight=0.5,
        entropy_weight=0.01,
        ppo_clip=0.2,
        vf_clip=10.0
    )

    run_offline(env, agent, episodes_per_learn=5, max_frames=150_000)


def run_mountain_car(visualize=False):
    from reinforcement_learning.run_offline import run_offline
    import gymnasium as gym

    env = gym.make("MountainCar-v0", render_mode="human" if visualize else None)
    # This episode length is necessary for learning correctly.
    env._max_episode_steps = 5000

    policy = SepCatMLP(
        n_features=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        actor_lr=1e-3,
        critic_lr=1e-3,
        d=128
    )

    agent = PPO(
        policy=policy,
        gamma=1.0,
        gae_lambda=0.95,
        ppo_epochs=3,
        batch_size=64,
        vf_weight=0.5,
        entropy_weight=0.01,
        ppo_clip=0.2,
        vf_clip=10.0
    )

    run_offline(env, agent, episodes_per_learn=5, max_frames=1_000_000)


def run_bipedal_walker(visualize=False):
    from reinforcement_learning.run_offline import run_offline
    import gymnasium as gym

    env = gym.make("BipedalWalker-v3", render_mode="human" if visualize else None)

    policy = SqueezeMLP(
        n_features=env.observation_space.shape[0],
        n_actions=4,
        low=-1.0, high=1.0,
        net_arch=[128]*3,
        actor_lr=1e-3,
        critic_lr=1e-3
    )
    # idea: Tanh activation, fixed variance

    agent = PPO(
        policy=policy,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=3,
        batch_size=128,
        vf_weight=0.5,
        entropy_weight=0.01,
        ppo_clip=0.2,
        vf_clip=100.0
    )

    run_offline(env, agent, episodes_per_learn=10, max_frames=2_000_000)


if __name__ == '__main__':
    run_bipedal_walker(visualize=False)
