"""Loading a saved policy, in one place.

It lives here rather than in the filming script because two callers need it and
the filming script is called `film.py`, which is also the name of the shared
recorder package at the repository root. A module that imports a package with
its own name resolves to whichever directory sys.path happens to list first —
and the error that produces, a partially initialised module, says nothing about
the real cause.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .policy import GaussianActorCritic, RunningNorm


def load(path: Path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = GaussianActorCritic(state["obs_dim"], state["act_dim"])
    model.load_state_dict(state["model"])
    model.eval()
    norm = RunningNorm((state["obs_dim"],))
    # The normaliser travels with the weights. A policy restored without it
    # reads its inputs in different units from the ones it trained on, which
    # looks exactly like a policy that has forgotten how to walk.
    norm.load_state_dict(state["norm"])
    return model, norm, state
