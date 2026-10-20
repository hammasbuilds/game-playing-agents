"""A Gaussian policy for continuous control.

Everything else in this repository chooses between a handful of buttons. A
MuJoCo robot does not have buttons: an action is a torque for every joint, six
or eight real numbers at once, and a softmax has nothing to say about that. So
the actor outputs a mean per joint and carries one learned log standard
deviation per joint — state-independent, which is the standard choice here and
the one that behaves on a CPU budget, because a state-dependent spread gives the
optimiser a second way to make the loss go down without walking any better.

The observation is a vector of joint angles and velocities, so there is no
convolution anywhere in this file. Two hidden layers of 64 with tanh is the
configuration every PPO paper uses for these tasks, and it is small enough that
the whole run fits on a laptop core.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def layer(in_dim: int, out_dim: int, gain: float = np.sqrt(2)) -> nn.Linear:
    """Orthogonal init, which is what keeps early PPO updates from exploding."""
    linear = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(linear.weight, gain)
    nn.init.constant_(linear.bias, 0.0)
    return linear


class GaussianActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64,
                 log_std_init: float = -0.5):
        super().__init__()
        self.actor = nn.Sequential(
            layer(obs_dim, hidden), nn.Tanh(),
            layer(hidden, hidden), nn.Tanh(),
            # A near-zero final layer starts the robot limp rather than
            # thrashing: the first thing a thrashing policy learns is that
            # moving hurts, and it then refuses to move at all.
            layer(hidden, act_dim, gain=0.01),
        )
        self.critic = nn.Sequential(
            layer(obs_dim, hidden), nn.Tanh(),
            layer(hidden, hidden), nn.Tanh(),
            layer(hidden, 1, gain=1.0),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        self.obs_dim, self.act_dim = obs_dim, act_dim

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(obs)
        return torch.distributions.Normal(mean, self.log_std.exp())

    def forward(self, obs: torch.Tensor):
        return self.actor(obs), self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, greedy: bool = False):
        dist = self.distribution(obs)
        action = dist.mean if greedy else dist.sample()
        logp = dist.log_prob(action).sum(-1)
        value = self.critic(obs).squeeze(-1)
        return action.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self.distribution(obs)
        # Summed over joints, because one action is the whole vector: the log
        # probability of a six-joint action is the sum of the six.
        logp = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy, self.critic(obs).squeeze(-1)


class RunningNorm:
    """Keeps observations near unit scale, updated from the data as it arrives.

    Joint angles sit in [-1, 1] and velocities reach double digits, so a network
    fed the raw vector spends its first thousand updates learning the scale of
    its own inputs. This is the one piece of machinery that continuous control
    needs and discrete pixel control does not — frames are already bounded.

    The statistics belong in the checkpoint: a policy restored without them is
    reading its inputs in different units from the ones it trained on, which
    looks like a policy that forgot how to walk.
    """

    def __init__(self, shape, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, batch: np.ndarray) -> None:
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / total) / total
        self.count = total

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.clip((obs - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0
                       ).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(),
                "count": float(self.count)}

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.array(state["mean"], dtype=np.float64)
        self.var = np.array(state["var"], dtype=np.float64)
        self.count = float(state["count"])
