"""The policy: a three-layer conv trunk with a value head, ~290k parameters.

Sized for a CPU budget rather than for a benchmark table. The trunk is the
Mnih et al. DQN trunk with the channel counts halved, which on this machine costs
about a third of the forward time of the full one and reaches the same score on
these scenarios.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def _ortho(layer: nn.Module, gain: float) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int, hidden: int = 256,
                 width: int = 1):
        """`width` scales every channel count and the hidden layer.

        width=1 is the CPU-sized trunk -- the Mnih et al. DQN trunk with the
        channels halved -- and is what the laptop runs. Wider trunks are only
        worth their forward pass on a GPU, which is what the Kaggle notebook in
        `kaggle/` exists to test.
        """
        super().__init__()
        c, h, w = obs_shape
        ch1, ch2, ch3 = 16 * width, 32 * width, 32 * width
        hidden = hidden * width
        self.width = width
        self.trunk = nn.Sequential(
            _ortho(nn.Conv2d(c, ch1, 8, stride=4), np.sqrt(2)), nn.ReLU(),
            _ortho(nn.Conv2d(ch1, ch2, 4, stride=2), np.sqrt(2)), nn.ReLU(),
            _ortho(nn.Conv2d(ch2, ch3, 3, stride=1), np.sqrt(2)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.trunk(torch.zeros(1, c, h, w)).shape[1]
        self.body = nn.Sequential(_ortho(nn.Linear(n_flat, hidden), np.sqrt(2)), nn.ReLU())
        # Small gain on the policy head keeps the opening policy near uniform, so
        # the first rollouts explore instead of committing to one button.
        self.pi = _ortho(nn.Linear(hidden, n_actions), 0.01)
        self.v = _ortho(nn.Linear(hidden, 1), 1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Observations arrive as uint8 straight from the frame buffers; scaling
        # here rather than in the env keeps the 4x-smaller dtype on the wire.
        if obs.dtype == torch.uint8:
            obs = obs.float().div_(255.0)
        z = self.body(self.trunk(obs))
        return self.pi(z), self.v(z).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, greedy: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = obs.to(next(self.parameters()).device, non_blocking=True)
        logits, value = self(obs)
        if greedy:
            action = logits.argmax(dim=-1)
        else:
            action = Categorical(logits=logits).sample()
        logp = Categorical(logits=logits).log_prob(action)
        return (action.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy())

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
