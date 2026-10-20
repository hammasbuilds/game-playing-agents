# Game playing agents

Eight games, each in its own folder, each with the agent that plays it, a short
demonstration and the numbers. Every game is a real one that was downloaded and
run unmodified — nothing here is a toy written for the occasion — and every
agent was trained on a laptop CPU.

| # | project | game | how it plays | result |
|---|---|---|---|---|
| 1 | [DOOM, from pixels](01-doom/) | DOOM (Freedoom data) | PPO written out rather than imported, trained from raw frames on CPU | **9.80** vs 1.23 (spray) — learned |
| 2 | [Crafter, planned](02-crafter/) | Crafter | tile decoding into a map, then a goal ladder — no training at all | **12.73** vs 1.13 Crafter score (%) — no error bar |
| 3 | [Brogue, through the screen](03-brogue/) | Brogue CE | screenshots in, keystrokes out — no API, no memory reading | **1.00** vs 1.33 descents in 300 keystrokes — inside noise |
| 4 | [A robot learns to walk](04-walker/) | MuJoCo Walker2d | PPO with a Gaussian policy — an action is a torque per joint | **4,498** vs 65 untrained — learned |
| 5 | [Car racing](05-racing/) | CarRacing-v3 (Box2D) | the shared arcade PPO: 84×84 greyscale, four stacked frames | **638** vs -38 untrained — learned |
| 6 | [Boxing (retired)](06-boxing/) | Atari 2600 Boxing | the shared arcade PPO | **2** vs -1 untrained — inside noise |
| 7 | [Kangaroo (retired)](07-kangaroo/) | Atari 2600 Kangaroo | the shared arcade PPO | **30** vs 50 untrained — no trained checkpoint beat the untrained network |
| 8 | [Table tennis (retired)](08-pong/) | Atari 2600 Pong | the shared arcade PPO | **-13** vs -20 untrained — learned |

Open any folder: it has a comparison still, an eight-second clip, the result
with its error bar, and the commands that produced both.

## How every number here is measured

One file, [`protocol.py`](protocol.py), decides what the numbers mean:

1. **The last checkpoint is not the trained agent.** The ladder is scored and a
   checkpoint is chosen — one run here peaked in the middle and got worse.
2. **Choosing is itself a measurement**, so it uses its own episodes. The winner
   is re-scored on a disjoint set, and that second number is the one reported.
   Quoting the score that won the selection reports luck as skill.
3. **A gain counts only above two standard errors of the difference.** A gap
   smaller than its own uncertainty is not a win, and is not written up as one.

That is why a result in this repository is sometimes *inside noise* — the Brogue
agent's, for one. Those rows are printed beside the wins rather than dropped.

## Where the games come from

| game | what it is | where it comes from |
|---|---|---|
| DOOM | the 1993 engine with free Freedoom data | `vizdoom` (pip) |
| Crafter | an open-source survival game, 2021 | `crafter` (pip) |
| Brogue CE | a GPL roguelike, the real Windows build | release download |
| CarRacing | Gymnasium's Box2D racer | `gymnasium[box2d]` (pip) |
| Walker2d | a MuJoCo robot | `mujoco` (pip) |
| Boxing · Kangaroo · Pong | Atari 2600, 1977–1982 | ROMs shipped under licence with `ale-py` |

No commercial ROMs, nothing patched, nothing emulated by hand. The games
themselves are not committed here: each one is a package install or a download.

The three Atari games are kept with their numbers but retired from the
showcase — a 1980 console at 160×210 cannot be made to look good beside a
640×480 DOOM render, however well the agent plays.

Built from [game-agents](https://github.com/hammasbuilds/game-agents), which
holds the same work arranged by machinery rather than by game, plus the live
dashboard and the control panel.
