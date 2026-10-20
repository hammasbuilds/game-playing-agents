"""The evaluation protocol every project in this repository is scored by.

Three rules, written once here instead of four times badly:

**1. The last checkpoint is not the trained agent.** Both long runs in this
repository peaked in the middle and got worse: the walker reached 4,502 at
700k steps and finished at 3,550 by 2M; racing peaked at 60k and declined for
the next 360k. Reporting the final checkpoint is reporting whichever point
training happened to stop at.

**2. Choosing the best checkpoint is itself a measurement, so it needs its own
seeds.** Picking the highest scorer on a set of episodes and then quoting that
same score is selection bias — with six checkpoints and noisy episodes, the
winner is partly the luckiest. So the ladder is scored on `SELECT_SEEDS`, the
winner is re-scored on the disjoint `REPORT_SEEDS`, and the reported number is
the second one.

**3. A gain is a gain only if it is bigger than the uncertainty of the
difference.** Two standard errors of the difference, not two standard
deviations: with eight episodes a hundred-point gap in a racing score can be
nothing at all.

Nothing here knows what a game is. A project supplies a callable that plays N
episodes from a seed list and returns the returns; everything else is shared.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

# Disjoint on purpose. Selection and reporting must not share an episode.
SELECT_SEEDS = tuple(range(2_000, 2_020))
REPORT_SEEDS = tuple(range(9_000, 9_020))


@dataclass
class Score:
    """What N episodes of one agent came to."""

    returns: list[float]
    seeds: list[int] = field(default_factory=list)
    lengths: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.returns)

    @property
    def mean(self) -> float:
        return float(np.mean(self.returns)) if self.returns else float("nan")

    @property
    def std(self) -> float:
        return float(np.std(self.returns, ddof=1)) if self.n > 1 else 0.0

    @property
    def se(self) -> float:
        """Standard error: how far the *mean* is likely to be from the truth.

        This, not the standard deviation, is what a comparison of means is
        allowed to use. An earlier version of this repository compared two means
        against the sum of their standard deviations and called real learning
        noise.
        """
        return self.std / np.sqrt(self.n) if self.n > 1 else 0.0

    def as_dict(self) -> dict:
        return {
            "episodes": self.n,
            "return_mean": round(self.mean, 2),
            "return_std": round(self.std, 2),
            "return_se": round(self.se, 2),
            "return_min": round(float(np.min(self.returns)), 2) if self.returns else None,
            "return_max": round(float(np.max(self.returns)), 2) if self.returns else None,
            "steps_mean": round(float(np.mean(self.lengths)), 1) if self.lengths else None,
            "seeds": list(self.seeds),
            "returns": [round(float(r), 2) for r in self.returns],
        }


def verdict(challenger: Score, baseline: Score, challenger_name: str = "trained",
            baseline_name: str = "untrained") -> dict:
    """Did the challenger beat the baseline by more than the measurement can explain?"""
    gain = challenger.mean - baseline.mean
    bar = 2 * float(np.hypot(challenger.se, baseline.se))
    if gain > bar:
        label = "learned"
    elif gain < -bar:
        label = f"worse than {baseline_name}"
    else:
        label = "inside noise"
    return {
        "challenger": challenger_name, "baseline": baseline_name,
        "challenger_mean": round(challenger.mean, 2),
        "baseline_mean": round(baseline.mean, 2),
        "gain": round(gain, 2), "bar": round(bar, 2), "label": label,
        "episodes": challenger.n,
    }


def _log(message: str) -> None:
    """Print, flushed. Scoring a ladder takes minutes and its output is usually
    redirected to a file, where an unflushed print means the log stays empty
    until the run ends and there is no way to tell it apart from a hang."""
    print(message, flush=True)


def score_ladder(checkpoints: Sequence, play: Callable[[object, Iterable[int]], Score],
                 select_seeds: Sequence[int] = SELECT_SEEDS,
                 report_seeds: Sequence[int] = REPORT_SEEDS,
                 baseline: tuple | None = None,
                 extra: Sequence | None = None,
                 log=_log) -> dict:
    """Score every checkpoint, pick the best, and re-score it on fresh episodes.

    `checkpoints` is a sequence of (label, handle) pairs, oldest first, and
    `play(handle, seeds)` returns a Score. These are the *candidates*: the best
    of them is chosen and reported.

    `baseline` is what the winner is compared against — by default the first
    checkpoint, which in every run here is the untrained network. It is passed
    explicitly because getting it wrong is silent: the arcade evaluation put a
    random policy first in the list, so a comparison that said "against the
    untrained network" was really against random play, and on Kangaroo the
    random policy won the selection and was reported as the best *checkpoint*.

    `extra` rows are scored and kept for context — a random floor belongs in the
    table and does not belong in the competition.
    """
    rows = []

    def score(label, handle, seeds):
        result = play(handle, seeds)
        log(f"  {str(label):<16} return {result.mean:>9.1f} ± {result.se:>6.1f} "
            f"over {result.n} episodes")
        return result

    base_label, base_handle = baseline if baseline is not None else checkpoints[0]
    candidate_list = [(label, handle) for label, handle in checkpoints
                      if label != base_label]
    if not candidate_list:
        raise ValueError("every checkpoint is the baseline; nothing to score")

    for label, handle in (extra or []):
        rows.append({"checkpoint": label, "candidate": False,
                     **score(label, handle, select_seeds).as_dict()})
    # The baseline is in the table but never in the race. It was a candidate
    # once, and on Kangaroo -- where training makes the agent worse -- the
    # untrained network won its own comparison, which reads as "inside noise"
    # and hides the finding: no trained checkpoint beat it.
    base_select = score(base_label, base_handle, select_seeds)
    rows.append({"checkpoint": base_label, "candidate": False,
                 "baseline": True, **base_select.as_dict()})
    for label, handle in candidate_list:
        rows.append({"checkpoint": label, "candidate": True,
                     **score(label, handle, select_seeds).as_dict()})

    candidates = [r for r in rows if r["candidate"]]
    best_row = max(candidates, key=lambda r: r["return_mean"])
    best_index = [c["checkpoint"] for c in candidates].index(best_row["checkpoint"])
    best_label, best_handle = candidate_list[best_index]

    # The two numbers that get quoted are measured on episodes no checkpoint was
    # chosen on.
    log(f"\n  best on the selection episodes: {best_label}")
    best_report = score(f"{best_label} (held out)", best_handle, report_seeds)
    base_report = score(f"{base_label} (held out)", base_handle, report_seeds)

    out = {
        "select_seeds": list(select_seeds),
        "report_seeds": list(report_seeds),
        "ladder": rows,
        "best": {"checkpoint": best_label, **best_report.as_dict()},
        "untrained": {"checkpoint": base_label, **base_report.as_dict()},
        "final": candidates[-1],
        "verdict": verdict(best_report, base_report, baseline_name=str(base_label)),
    }
    # The plainest fact a ladder can produce, and the one a bare verdict loses:
    # not one of the trained checkpoints was better than the network they all
    # started from.
    out["none_beat_the_baseline"] = bool(
        best_row["return_mean"] <= base_select.mean)
    if out["none_beat_the_baseline"]:
        log(f"  no checkpoint beat {base_label} on the selection episodes "
            f"(best candidate {best_row['return_mean']:.1f} against "
            f"{base_select.mean:.1f})")
    # Whether training should have stopped earlier is a fact about the run, and
    # it is the one a "final checkpoint" report hides. But a later checkpoint
    # being nominally lower is not a peak: Pong's best was two checkpoints from
    # the end by a margin of zero, and the page dutifully said "peaked mid-run
    # and lost 0". The drop has to clear the uncertainty of the two rows being
    # compared, like every other claim here.
    final = candidates[-1]
    peak = candidates[best_index]
    drop = peak["return_mean"] - final["return_mean"]
    drop_bar = 2 * float(np.hypot(peak["return_se"], final["return_se"]))
    out["decline_after_peak"] = round(drop, 2)
    out["decline_bar"] = round(drop_bar, 2)
    out["peaked_early"] = bool(best_index < len(candidates) - 1 and drop > drop_bar)
    if out["peaked_early"]:
        log(f"  peaked at {best_label} and declined {drop:.1f} by the end of "
            f"training, against a bar of {drop_bar:.1f}")
    elif best_index < len(candidates) - 1:
        log(f"  best was {best_label}, {drop:.1f} above the final checkpoint "
            f"— inside a bar of {drop_bar:.1f}, so not a decline")
    return out


# -- the record of rounds ----------------------------------------------------
def record_round(entry: dict, path=None) -> None:
    """Append one scored round to the repository's round log.

    A single result says what an agent does; the log says what *trying again*
    did, which is the only way to tell a real improvement from a seed. Each line
    carries the configuration that produced it, so a round that beat the last
    one can be repeated rather than admired.

    Appending, never rewriting: a round whose numbers came out badly stays in
    the file. The whole value of the log is that it contains the attempts that
    did not work.
    """
    import json
    import subprocess
    from datetime import datetime, timezone
    from pathlib import Path

    root = Path(__file__).resolve().parent
    path = Path(path) if path else root / "docs" / "data" / "rounds.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    row = {"at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
           "commit": commit or None, **entry}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + chr(10))
