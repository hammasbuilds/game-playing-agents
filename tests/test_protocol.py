"""The evaluation protocol: the part that decides what every number means.

Each of these pins a mistake this repository actually made. Comparing means
against standard deviations called real learning noise. Reporting the last
checkpoint called a run that got worse a success. Quoting the score that won a
selection is how a noisy ladder reports luck as skill.
"""
from __future__ import annotations

import numpy as np
import pytest

from protocol import REPORT_SEEDS, SELECT_SEEDS, Score, score_ladder, verdict


def test_selection_and_reporting_seeds_are_disjoint():
    """The whole point: no checkpoint may be chosen on an episode it is scored on."""
    assert not set(SELECT_SEEDS) & set(REPORT_SEEDS)


def test_standard_error_is_not_standard_deviation():
    score = Score(returns=[10.0, 20.0, 30.0, 40.0])
    assert score.std == pytest.approx(12.909944, rel=1e-5)
    assert score.se == pytest.approx(score.std / 2, rel=1e-9)   # sqrt(4) = 2


def test_a_gain_inside_the_noise_is_not_a_win():
    """Boxing: +0.5 of mean against spread that explains it."""
    trained = Score(returns=[2.0, 0.0, 4.0, -2.0, 6.0])
    untrained = Score(returns=[1.5, 0.5, 3.5, -1.5, 5.0])
    out = verdict(trained, untrained)
    assert out["label"] == "inside noise"


def test_a_real_gain_is_called_one():
    trained = Score(returns=[300.0, 310.0, 295.0, 305.0])
    untrained = Score(returns=[-40.0, -50.0, -45.0, -42.0])
    assert verdict(trained, untrained)["label"] == "learned"


def test_going_backwards_is_named():
    """The platformer collapsed onto one button and ended below its own start."""
    trained = Score(returns=[0.0, 0.0, 0.0, 0.0])
    untrained = Score(returns=[25.0, 30.0, 20.0, 25.0])
    assert verdict(trained, untrained)["label"] == "worse than untrained"


def fake_ladder(select_values, report_values):
    """A play() whose answer depends on which seed set it is handed."""
    def play(handle, seeds):
        table = report_values if set(seeds) & set(REPORT_SEEDS) else select_values
        value = table[handle]
        # A little spread so the standard error is not zero.
        return Score(returns=[value - 1, value, value + 1], seeds=list(seeds))
    return play


def test_the_reported_score_comes_from_the_held_out_episodes():
    """This is the anti-selection-bias property, and it has to be provable.

    Checkpoint 2 wins on the selection episodes by luck and is worth much less
    on fresh ones. The reported number must be the fresh one, or the ladder
    reports its own luck.
    """
    checkpoints = [("ckpt_000", 0), ("ckpt_001", 1), ("ckpt_002", 2)]
    play = fake_ladder(select_values={0: 10.0, 1: 50.0, 2: 90.0},
                       report_values={0: 10.0, 1: 48.0, 2: 20.0})
    out = score_ladder(checkpoints, play, select_seeds=SELECT_SEEDS[:3],
                       report_seeds=REPORT_SEEDS[:3], log=lambda *_: None)
    assert out["best"]["checkpoint"] == "ckpt_002"
    assert out["best"]["return_mean"] == pytest.approx(20.0)
    assert out["ladder"][2]["return_mean"] == pytest.approx(90.0)


def test_a_run_that_peaked_early_says_so():
    checkpoints = [("ckpt_000", 0), ("ckpt_001", 1), ("ckpt_002", 2)]
    play = fake_ladder(select_values={0: 10.0, 1: 90.0, 2: 40.0},
                       report_values={0: 10.0, 1: 88.0, 2: 41.0})
    out = score_ladder(checkpoints, play, select_seeds=SELECT_SEEDS[:3],
                       report_seeds=REPORT_SEEDS[:3], log=lambda *_: None)
    assert out["peaked_early"] is True
    assert out["decline_after_peak"] == pytest.approx(50.0)
    assert out["best"]["checkpoint"] == "ckpt_001"


def test_a_run_that_ended_at_its_best_does_not_claim_a_peak():
    checkpoints = [("ckpt_000", 0), ("ckpt_001", 1), ("ckpt_002", 2)]
    play = fake_ladder(select_values={0: 10.0, 1: 40.0, 2: 90.0},
                       report_values={0: 10.0, 1: 41.0, 2: 88.0})
    out = score_ladder(checkpoints, play, select_seeds=SELECT_SEEDS[:3],
                       report_seeds=REPORT_SEEDS[:3], log=lambda *_: None)
    assert out["peaked_early"] is False
    # The drop is always reported -- it is a measurement. Only the claim that
    # the run peaked is conditional on it clearing its own uncertainty.
    assert out["decline_after_peak"] <= 0


def test_the_first_checkpoint_is_the_baseline():
    """Every run here keeps step zero; the verdict is measured against it."""
    checkpoints = [("ckpt_000", 0), ("ckpt_001", 1)]
    play = fake_ladder(select_values={0: 5.0, 1: 95.0},
                       report_values={0: 4.0, 1: 96.0})
    out = score_ladder(checkpoints, play, select_seeds=SELECT_SEEDS[:3],
                       report_seeds=REPORT_SEEDS[:3], log=lambda *_: None)
    assert out["untrained"]["checkpoint"] == "ckpt_000"
    assert out["verdict"]["baseline_mean"] == pytest.approx(4.0)
    assert out["verdict"]["label"] == "learned"


def test_a_floor_does_not_compete_for_best():
    """A random policy belongs in the table and not in the race.

    The arcade evaluation put one in the candidate list. On Kangaroo, where no
    checkpoint beats random play, the random policy won the selection and was
    reported as the best *checkpoint* — a sentence that cannot be true.
    """
    checkpoints = [("ckpt_000", 0), ("ckpt_001", 1)]
    play = fake_ladder(select_values={0: 10.0, 1: 20.0, "rand": 90.0},
                       report_values={0: 10.0, 1: 21.0, "rand": 88.0})
    out = score_ladder(checkpoints, play, select_seeds=SELECT_SEEDS[:3],
                       report_seeds=REPORT_SEEDS[:3],
                       extra=[("random", "rand")], log=lambda *_: None)
    assert out["best"]["checkpoint"] == "ckpt_001"
    # The floor is still in the table, where a reader can see it beat everything.
    floors = [r for r in out["ladder"] if r["checkpoint"] == "random"]
    assert floors and floors[0]["return_mean"] == pytest.approx(90.0)
    assert floors[0]["candidate"] is False


def test_the_baseline_is_whatever_was_passed():
    """Saying "against the untrained network" must mean that network.

    With the baseline taken from position zero, putting a random policy first in
    the list silently changed what every verdict in the arcade was measured
    against — and the word "untrained" in the output did not change with it.
    """
    checkpoints = [("ckpt_000", 0), ("ckpt_001", 1)]
    play = fake_ladder(select_values={0: 10.0, 1: 50.0, "rand": 30.0},
                       report_values={0: 11.0, 1: 51.0, "rand": 29.0})
    out = score_ladder(checkpoints, play, select_seeds=SELECT_SEEDS[:3],
                       report_seeds=REPORT_SEEDS[:3],
                       baseline=("ckpt_000", 0), extra=[("random", "rand")],
                       log=lambda *_: None)
    assert out["untrained"]["checkpoint"] == "ckpt_000"
    assert out["verdict"]["baseline_mean"] == pytest.approx(11.0)
    assert out["verdict"]["baseline"] == "ckpt_000"


def test_peaked_early_ignores_the_floor_rows():
    """Context rows must not shift which candidate counts as the last one."""
    checkpoints = [("ckpt_000", 0), ("ckpt_001", 1), ("ckpt_002", 2)]
    play = fake_ladder(select_values={0: 10.0, 1: 90.0, 2: 40.0, "rand": 5.0},
                       report_values={0: 10.0, 1: 88.0, 2: 41.0, "rand": 5.0})
    out = score_ladder(checkpoints, play, select_seeds=SELECT_SEEDS[:3],
                       report_seeds=REPORT_SEEDS[:3],
                       baseline=("ckpt_000", 0), extra=[("random", "rand")],
                       log=lambda *_: None)
    assert out["peaked_early"] is True
    assert out["decline_after_peak"] == pytest.approx(50.0)


def test_the_baseline_cannot_win_its_own_comparison():
    """Kangaroo: every trained checkpoint is worse than the network it started
    from. With the baseline in the race it won, and the verdict read "inside
    noise" — which is true of a thing compared with itself and tells nobody
    that training made the agent worse."""
    checkpoints = [("ckpt_000", 0), ("ckpt_001", 1), ("ckpt_002", 2)]
    play = fake_ladder(select_values={0: 60.0, 1: 10.0, 2: 0.0},
                       report_values={0: 55.0, 1: 9.0, 2: 0.0})
    out = score_ladder(checkpoints, play, select_seeds=SELECT_SEEDS[:3],
                       report_seeds=REPORT_SEEDS[:3],
                       baseline=("ckpt_000", 0), log=lambda *_: None)
    assert out["best"]["checkpoint"] == "ckpt_001"       # best TRAINED one
    assert out["untrained"]["checkpoint"] == "ckpt_000"
    assert out["none_beat_the_baseline"] is True
    rows = {r["checkpoint"]: r for r in out["ladder"]}
    assert rows["ckpt_000"]["candidate"] is False        # in the table, not the race


def test_a_run_where_something_did_beat_it_says_nothing_of_the_sort():
    checkpoints = [("ckpt_000", 0), ("ckpt_001", 1)]
    play = fake_ladder(select_values={0: 10.0, 1: 90.0},
                       report_values={0: 11.0, 1: 88.0})
    out = score_ladder(checkpoints, play, select_seeds=SELECT_SEEDS[:3],
                       report_seeds=REPORT_SEEDS[:3],
                       baseline=("ckpt_000", 0), log=lambda *_: None)
    assert out["none_beat_the_baseline"] is False


def test_a_drop_inside_the_noise_is_not_a_peak():
    """Pong's best checkpoint was two from the end by a margin of zero, and the
    page said "peaked mid-run and lost 0". A decline is a claim like any other
    and has to clear the uncertainty of the two rows being compared."""
    checkpoints = [("ckpt_000", 0), ("ckpt_001", 1), ("ckpt_002", 2)]
    play = fake_ladder(select_values={0: 10.0, 1: 50.0, 2: 49.9},
                       report_values={0: 10.0, 1: 50.0, 2: 49.9})
    out = score_ladder(checkpoints, play, select_seeds=SELECT_SEEDS[:3],
                       report_seeds=REPORT_SEEDS[:3],
                       baseline=("ckpt_000", 0), log=lambda *_: None)
    assert out["best"]["checkpoint"] == "ckpt_001"
    assert out["peaked_early"] is False
    assert out["decline_after_peak"] == pytest.approx(0.1)
    assert out["decline_bar"] > out["decline_after_peak"]


def test_a_real_decline_still_counts_as_one():
    checkpoints = [("ckpt_000", 0), ("ckpt_001", 1), ("ckpt_002", 2)]
    play = fake_ladder(select_values={0: 10.0, 1: 900.0, 2: 100.0},
                       report_values={0: 10.0, 1: 890.0, 2: 101.0})
    out = score_ladder(checkpoints, play, select_seeds=SELECT_SEEDS[:3],
                       report_seeds=REPORT_SEEDS[:3],
                       baseline=("ckpt_000", 0), log=lambda *_: None)
    assert out["peaked_early"] is True
    assert out["decline_after_peak"] == pytest.approx(800.0)
