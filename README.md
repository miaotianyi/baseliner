# baseliner
Train, test, and compare any machine learning model on every dataset in one click.

If your model performs well against all the baselines,
there are usually no bugs in the implementation.

## Contributions
To contribute, just tell us what downstream task you want incorporated
into `baseliner`.

Currently, we're working on reinforcement learning
and tabular datasets (coming soon).

## Reinforcement Learning
Suppose you've written a reinforcement learning algorithm
but it performs worse than expected.
How can we know which part goes wrong?
Is it because the environment is inherently difficult,
or is there an implementation bug in the RL algorithm?
A good RL algorithm should perform well across many environments.
So we run it against many classic environments (`gym`) in one click.

Many real-world environments don't expose an `env.step(action)` API in Python.
In many cases, we have to deploy the environment in one (Windows) computer
and train the agent in another (GPU/Linux) server, so offline training is required.
So the agent interface is:
```python
class Agent:
    def act(self, obs):
        # obs is observation at this timestep
        action = None
        return action

    def see(self, reward, next_obs):
        pass

    def learn(self, episodes):
        # each episode is a (obs_list, action_list, reward_list)
        pass
```

