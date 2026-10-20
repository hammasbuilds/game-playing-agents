# The arcade — one network, three games, and the gap between step 1 and the last step

Three visual games, one training stack, and a deliberate output: a video in which
the **untrained network and the trained one play the same level at the same
time**. The point of this half of the repository is not a leaderboard number. It
is being able to see what learning did.

| game | what it is | why it is here |
|---|---|---|
| **Car racing** | Gymnasium `CarRacing-v3`, discrete actions | the improvement is visible without reading a score: an untrained car spins on the grass, a trained one follows the road |
| **Boxing** | `ALE/Boxing-v5` | hand to hand; the score is punches landed minus punches taken, so it runs from −100 to +100 |
| **Platformer** | `ALE/Kangaroo-v5` | climb, punch, reach the top — the legal stand-in for Mario, since `ale-py` ships its ROMs under licence and NES ROMs are not mine to distribute |

Nothing here needs a ROM you have to find yourself, and nothing is specialised to
one game: every one is reduced to a stack of four 84×84 greyscale frames and
learned by the same network with the same PPO.

## The thing this is built around

**Checkpoints are written from step 0** — before a single gradient step — and
then at 5%, 15%, 35%, 65% and 100% of training. That is not bookkeeping. A
before-and-after comparison only exists if somebody saved the "before", and the
untrained network is the most useful baseline there is: it is the same
architecture, the same observation, the same action set, with nothing learned.

`arcade/progress.py` then does two things with those files:

1. **Scores every checkpoint on the same evaluation levels.** Same seeds for all
   of them, so the climb is like for like rather than a tour of different levels.
   A random-key policy is scored on those levels too, as the floor.
2. **Films them playing the same seed**, and lays the clips out side by side into
   one autoplaying video, each panel labelled with how long that network had
   trained and what it scored.

```bash
python -m arcade.train --all --envs 12      # trains every game, saving checkpoints
python -m arcade.progress --game boxing     # ladder + side-by-side film
python ui/export_arcade.py                  # feed it to the dashboard
```

Results land in `docs/media/<game>/` and appear on the dashboard as a card per
game; the card is a button to that game's own page.

## What an hour of CPU buys, per game

| game | untrained | trained | verdict |
|---|---:|---:|---|
| **Car racing** | −53.7 | **+376.0** at 419k steps | learned — +429.7 against a bar of 130.2 |
| **Table tennis** | −20.0 | −16.6 at 2M steps | learned — +3.4 against a bar of 2.1 |
| Boxing | 2.12 | 1.62 at 2M steps | **no gain** — −0.5 against a bar of 3.4 |
| Platformer | 25.0 | 0.0 at 2M steps | **worse than untrained** — see below |

Racing's own ladder, every row on the same six levels:

| checkpoint | steps | score |
|---|---:|---:|
| random keys | — | −50.4 |
| untrained | 0 | −53.7 |
| 5% | 59,904 | +135.0 |
| 15% | 179,712 | +300.8 |
| **35%** | **419,328** | **+376.0** |

Every row scores the same levels for every checkpoint, against a random-key
policy and against the untrained network. The bar is two standard errors of the
difference, not two standard deviations — see below for why that distinction
decided one of these verdicts.

Racing is the one to look at first, because its score needs no explaining: a
negative number in CarRacing means the car mostly sat on the grass collecting the
per-step penalty. The untrained panel drifts off the road at −42 while the
60k-step panel tracks the curve at +156, on the same seed, at the same moment.

## A game where training made it worse, and why

Kangaroo is the most instructive run here, because it did not merely fail to
improve — it got **worse than the network that had never been trained**, and then
stayed there:

| checkpoint | steps | score over 8 episodes |
|---|---:|---:|
| random keys | — | 50.0 ± 92.6 |
| untrained | 0 | 25.0 ± 70.7 |
| 5% | 99,840 | **0.00 ± 0.00** |
| 15% | 299,520 | **0.00 ± 0.00** |
| 35% | 700,416 | **0.00 ± 0.00** |
| 65% | 1,299,456 | **0.00 ± 0.00** |
| 100% | 1,999,872 | **0.00 ± 0.00** |

A score of exactly zero with exactly zero spread, five checkpoints running, is
not a result to write down and move past — it is either a real collapse or a
broken evaluation, and those need telling apart. Counting the actions each
checkpoint actually took settles it:

```
ckpt 000 (untrained)  479 steps   top actions: 14 x35, 3 x29, 0 x28     spread out
ckpt 065 (trained)    479 steps   top actions: 10 x377, 7 x30, 14 x14   one button
```

The trained policy pressed a single button for 377 of 479 steps. It collapsed
onto a deterministic action that never scores, and the evaluation is fine — the
random and untrained arms score through the same code path.

**Why this game and not the others.** Kangaroo's first points need a specific
sequence — climb a ladder, punch a monkey, climb again — that random exploration
almost never completes, so there is nothing for the gradient to reinforce. With
no signal pulling it anywhere, the entropy bonus is the only thing keeping the
policy spread out, and at 0.01 it loses: the policy narrows, exploration stops,
and the collapse is self-sealing. Racing, by contrast, pays something for every
metre of road, so there is a gradient from the first episode.

This is the honest shape of the answer to "what does an hour of CPU buy": on a
game with dense reward, a car that learned to drive; on a game with sparse
reward, a policy worse than the one you started with.

## A game that did not learn, and how we know

Boxing trained for 2,000,000 steps — 56 minutes — and came out **no better than
the network that had never been trained at all**:

| checkpoint | score over the same 8 levels |
|---|---:|
| random keys | 0.50 ± 5.83 |
| untrained | 2.12 ± 3.27 |
| 5% | 3.00 ± 4.41 |
| 35% | 2.12 ± 6.31 |
| 65% | −2.12 ± 5.00 |
| **100%** | **1.62 ± 3.50** |

The ladder does not climb; it wanders inside its own error bars. The training
return said the same thing more quietly — it went from −0.25 to 2.55, which looks
like progress until you notice that the untrained network already scores 2.12.

This is the expected result rather than a broken run: published PPO curves put
Boxing near zero until about 5M steps and only reach a strong score around 10M,
so a laptop hour buys the flat part of the curve. It is kept here, with its
numbers, because a page that only shows the games that worked teaches the wrong
lesson about what an hour of training buys.

An earlier check of two checkpoints over **four** episodes read −2.5 against
+1.25 and looked like clear learning. It was noise. Eight episodes and a random
baseline were enough to see that; four were not.

**The sensitivity check that makes the null result mean something.** "No
improvement" is only evidence if the measurement could have found improvement.
The same pipeline, unchanged, was run on Pong over the same eight-episode
protocol:

| game | untrained | trained | verdict |
|---|---:|---:|---|
| Boxing | 2.12 | 1.62 | **no gain** — −0.5 against a bar of 3.4 |
| Pong | −20.00 | −16.62 | **learned** — +3.4 against a bar of 2.1 |

So the harness finds a climb where one exists and refuses one where it does not.
A null result on Boxing beside a clear positive on Pong says something about
Boxing's budget; a null result on its own would only say something about the
harness.

**The bar is two standard errors, not two standard deviations** — and getting
that wrong is what the first version of the verdict did. A standard deviation
says how much one episode differs from another; a standard error says how well
the *mean* is pinned down, and shrinks with the square root of the episode count.
Judged by deviations, Pong's +3.4 fell under a bar of 3.9 and was declared noise,
while its training curve had climbed from −21 to −14 over a hundred episodes. The
test has to match the quantity being compared.

## Two details that are easy to get wrong

**The untrained agent is filmed sampling, not taking the argmax.** A freshly
initialised policy is nearly uniform, and its argmax is one arbitrary action
repeated forever — a car that drives into a wall and stays there. That looks like
a broken program rather than like an untrained agent, and it would make the
"before" side of every comparison dishonest in the flattering direction.

**CarRacing's first fifty frames are a camera zoom** in which the car is a few
pixels across on an empty field. They are skipped before the episode is handed
over, because training on them teaches the policy to act on a picture it will
never see again. The reward collected during the skip is kept, so the episode
score still means what the game says it means.

## Cost

Everything runs on the CPU — the GPU on this machine belongs to another job. The
rollout, not the gradient, is the bottleneck in all three games, which is the
same thing the DOOM project found and the Kaggle notebook measured: four times
the network width costs 2% of the throughput.
