"""
We use "policy" to denote the neural network architecture (nn.Module)
and "agent" to denote the training loop boilerplate.
"""
import copy

import numpy as np
import torch
from torch import nn, optim


def gae(rewards: np.ndarray, critic_values: np.ndarray, gamma: float, gae_lambda: float):
    # Generalized advantage estimation
    # GAE[lambda](t) = ReturnsTD[lambda](t) - EstimatedValues(t)
    # The same gae_lambda is used for GAE and TD-lambda returns
    # Typical setting: gamma=0.99, gae_lambda=0.95
    n_steps = len(rewards)
    advantages = np.zeros(n_steps)
    advantages[-1] = rewards[-1] - critic_values[-1]
    for t in reversed(range(n_steps - 1)):  # from n_steps-2 to 0 inclusive
        delta = rewards[t] + gamma * critic_values[t + 1] - critic_values[t]
        advantages[t] = delta + gamma * gae_lambda * advantages[t + 1]
    return advantages


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
        return dist.sample()

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
            lr=self.base_lr, eps=1e-5)
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
        self.old_policy = copy.deepcopy(policy)
        # old policy is always in no-grad and eval mode.
        # These 2 settings will survive updates like
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
            logits = self.old_policy.sample(self.current_obs).unsqueeze(0)
            action = torch.distributions.Categorical(logits=logits).sample().item()
        return action

    def collate(self, episodes):
        # old values, old returns, old advantages, old log prob
        pass

