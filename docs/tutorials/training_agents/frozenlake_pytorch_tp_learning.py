"""
Solving Frozenlake with Tabular Q-Learning
==========================================

This tutorial trains an agent for FrozenLake using tabular Q-learning.
"""

# %%
# In this post we'll compare a bunch of different map sizes on the
# `FrozenLake <https://gymnasium.farama.org/environments/toy_text/frozen_lake/>`__
# environment from the reinforcement learning
# `Gymnasium <https://gymnasium.farama.org/>`__ package using the
# Q-learning algorithm.

# %%
# Let's first import a few dependencies we'll need.
#

# Author: Andrea Pierré
# License: MIT License

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import torch

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


sns.set_theme()

# %load_ext lab_black


# %%
# Parameters we'll use
# --------------------
#


class Params(NamedTuple):
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    proba_frozen: float  # Probability that a tile is frozen


params = Params(
    total_episodes=2000,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.1,
    map_size=5,
    seed=123,
    is_slippery=False,
    n_runs=20,
    action_size=None,
    state_size=None,
    proba_frozen=0.9,
)
params

# Set the seed
rng = np.random.default_rng(params.seed)

# %%
# The FrozenLake environment
# --------------------------
#

env = gym.make(
    "FrozenLake-v1",
    is_slippery=params.is_slippery,
    render_mode="rgb_array",
    desc=generate_random_map(
        size=params.map_size, p=params.proba_frozen, seed=params.seed
    ),
)

# %%

params = params._replace(action_size=env.action_space.n)
params = params._replace(state_size=env.observation_space.n)
print(f"Action size: {params.action_size}")
print(f"State size: {params.state_size}")


class SarSaLearning:
    def __init__(self, learning_rate, gamma, state_size, action_size, num_params, env):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.env = env
        self.action_space = env.action_space

    def init_network(self):
        """Return the Q-table."""
        total_sa_pairs = (
            self.env.unwrapped.observation_space.n * self.env.unwrapped.action_space.n
        )

        self.l1 = torch.nn.linear(
            in_features=total_sa_pairs,
            out_features=1,
            bias=True,
        )
        self.act = torch.nn.ReLU()

    def forward(self, s, a):
        """Forward pass of the network."""
        x = torch.nn.functional.one_hot(
            torch.LongTensor(s * self.action_size + a),
        )
        x = self.l1(x)
        x = self.act(x)
        return x

    def update(self, s, a, s_next, a_next, reward, done):
        """We need a specific backward pass for the SarSaLearning algorithm."""
        # first let's calculate the gradient of the network output with respect to the the weights
        # of the network
        curr_value = self.forward(s, a)
        gradient = curr_value.backward()
        if done:
            self.l1.weight = (
                self.l1.weight + self.learning_rate * (reward + curr_value) * gradient
            )
        else:
            # if we are not done, we need to update the weights of the network
            # with the gradient of the Q-value for the next state
            # and the Q-value for the current state
            self.l1.weight = (
                self.l1.weight
                + self.learning_rate
                * (reward + self.forward(s_next, a_next) - curr_value)
                * gradient
            )


class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose_action(self, action_space, state, sarsa):
        """Choose an action `a` in the current world state (s)."""
        # First we randomize a number
        explor_exploit_tradeoff = rng.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

        # Exploitation (taking the biggest state-action value for this state)
        else:
            action_values = {
                i: sarsa.forward(state, i) for i in range(sarsa.action_size)
            }
            # We don't really care in case of ties, we just take the first one
            action = max(action_values, key=action_values.get)
        return action
