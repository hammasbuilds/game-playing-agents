"""Retiring a game must not quietly delete its result.

Two of the three retired games scored badly — boxing gained nothing, the
platformer ended worse than its untrained checkpoint. Those are exactly the
results a showcase is tempted to lose, so the rule is written down here: a
retired game keeps its entry, keeps its reason, and still reaches the page.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arcade.envs import CATALOGUE, retired, showcase

ROOT = Path(__file__).resolve().parents[1]


def test_showcase_and_retired_partition_the_catalogue():
    assert set(showcase()) | set(retired()) == set(CATALOGUE)
    assert not set(showcase()) & set(retired())


def test_every_retired_game_says_why():
    assert retired(), "nothing is retired; this test is about what happens when it is"
    for key in retired():
        reason = CATALOGUE[key].retired
        assert reason and len(reason) > 20, key


def test_a_retired_game_is_still_playable_and_still_trainable():
    """Retirement is a presentation decision, not a deletion."""
    for key in retired():
        game = CATALOGUE[key]
        assert game.env_id and isinstance(game.kwargs, dict)


def test_the_exported_page_data_carries_retired_games_and_their_numbers():
    data = ROOT / "docs" / "data" / "arcade.json"
    if not data.exists():
        pytest.skip("run python -m ui.export_arcade first")
    games = json.loads(data.read_text())["games"]
    for key in retired():
        assert key in games, f"{key} vanished from the page data"
        assert games[key]["retired"], f"{key} is on the page with no reason given"
        ladder = games[key].get("ladder") or []
        assert ladder, f"{key} kept no scores; retiring it threw its result away"
