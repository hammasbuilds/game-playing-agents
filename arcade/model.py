"""The networks the arcade can train, and the factory that names them.

Three architectures, all reading the same 84x84 four-frame stack and all ending
in a policy head and a value head, so any of them can be dropped into the same
PPO without touching it:

* **nature** — the Mnih et al. DQN trunk, three convolutions and a hidden layer.
  The default, and what every result in this repository was measured with.
* **impala** — the residual trunk from IMPALA: three stages of convolution,
  max-pool and two residual blocks. More capacity per pixel and noticeably slower
  on a CPU; it is here because "would a better network help?" is a question worth
  being able to answer by changing a dropdown rather than by editing a file.
* **small** — half of nature again. For when the rollout is the bottleneck and
  you want the gradient out of the way entirely.

`width` multiplies the channel counts of whichever one is chosen, so the two
knobs compose. The Kaggle notebook measured what width costs: on a GPU, four
times the channels scores 10.43 against 8.53 on identical steps for 2% of the
throughput. On a CPU that trade runs the other way.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

ARCHITECTURES = ("nature", "impala", "small")


def _ortho(layer: nn.Module, gain: float) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ResidualBlock(nn.Module):
    """Two convolutions added back onto their own input."""

    def __init__(self, channels: int):
        super().__init__()
        self.a = _ortho(nn.Conv2d(channels, channels, 3, padding=1), np.sqrt(2))
        self.b = _ortho(nn.Conv2d(channels, channels, 3, padding=1), np.sqrt(2))

    def forward(self, x):
        y = self.a(torch.relu(x))
        y = self.b(torch.relu(y))
        return x + y


def _nature_trunk(c: int, width: int, small: bool) -> nn.Sequential:
    base = 8 if small else 16
    ch1, ch2, ch3 = base * width, 2 * base * width, 2 * base * width
    return nn.Sequential(
        _ortho(nn.Conv2d(c, ch1, 8, stride=4), np.sqrt(2)), nn.ReLU(),
        _ortho(nn.Conv2d(ch1, ch2, 4, stride=2), np.sqrt(2)), nn.ReLU(),
        _ortho(nn.Conv2d(ch2, ch3, 3, stride=1), np.sqrt(2)), nn.ReLU(),
        nn.Flatten(),
    )


def _impala_trunk(c: int, width: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    channels = c
    for out in (16 * width, 32 * width, 32 * width):
        layers += [
            _ortho(nn.Conv2d(channels, out, 3, padding=1), np.sqrt(2)),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResidualBlock(out),
            ResidualBlock(out),
        ]
        channels = out
    layers += [nn.ReLU(), nn.Flatten()]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """One network for every game in the arcade."""

    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int,
                 hidden: int = 256, width: int = 1, arch: str = "nature"):
        super().__init__()
        if arch not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {arch!r}; have {ARCHITECTURES}")
        c, h, w = obs_shape
        self.arch = arch
        self.width = width
        if arch == "impala":
            self.trunk = _impala_trunk(c, width)
        else:
            self.trunk = _nature_trunk(c, width, small=(arch == "small"))
        hidden = max(32, (hidden // (2 if arch == "small" else 1)) * width)
        with torch.no_grad():
            n_flat = self.trunk(torch.zeros(1, c, h, w)).shape[1]
        self.body = nn.Sequential(_ortho(nn.Linear(n_flat, hidden), np.sqrt(2)), nn.ReLU())
        # A small gain on the policy head keeps the opening policy near uniform,
        # so the first rollouts explore instead of committing to one button.
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
    def act(self, obs: torch.Tensor, greedy: bool = False):
        obs = obs.to(next(self.parameters()).device, non_blocking=True)
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if greedy else dist.sample()
        return (action.cpu().numpy(), dist.log_prob(action).cpu().numpy(),
                value.cpu().numpy())

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build(obs_shape, n_actions: int, arch: str = "nature", width: int = 1) -> ActorCritic:
    return ActorCritic(obs_shape, n_actions, width=width, arch=arch)
