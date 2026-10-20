"""Record the untrained robot and the trained one walking the same episode.

    python film.py --env Walker2d-v5 --seed 3

MuJoCo renders at whatever size it is asked for, so there is no reason for this
film to be small. The policy sees a vector of joint angles either way — the
camera and the agent share nothing here, which is exactly the point the DOOM
work had to be rebuilt to achieve.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from film import side_by_side, write_clip                        # noqa: E402
from walker.checkpoint import load                               # noqa: E402


def follow_the_robot(env, body: str = "torso", distance: float = 5.0) -> None:
    """Point the camera at the robot and keep it there.

    Two things have to be right, and the first film of a walking agent had
    neither: it showed an empty floor from halfway through, because the robot
    walked out of shot while the camera stayed at the origin.

    * Gymnasium sets `trackbodyid` in its default camera config but leaves the
      camera *type* free, so the setting is read and ignored.
    * `MujocoRenderer.render` passes its own `camera_id` to the viewer on every
      frame, and a camera_id of -1 resets the type back to free. Setting the
      camera once and rendering does nothing; the renderer undoes it each frame.
    """
    import mujoco
    renderer = getattr(env.unwrapped, "mujoco_renderer", None)
    if renderer is None:
        return
    renderer.camera_id = None                 # stop it overriding the type
    body_id = mujoco.mj_name2id(env.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, body)
    config = {"type": int(mujoco.mjtCamera.mjCAMERA_TRACKING),
              "trackbodyid": body_id, "distance": distance, "elevation": -12.0}
    renderer.default_cam_config = config      # for a viewer built later
    viewer = renderer._get_viewer("rgb_array")
    for key, value in config.items():
        setattr(viewer.cam, key, value)


def record(model, norm, env_id: str, seed: int, size: int, max_steps: int = 1000):
    env = gym.make(env_id, render_mode="rgb_array", width=size, height=size)
    obs, _ = env.reset(seed=seed)
    first = env.render()                      # builds the viewer
    follow_the_robot(env)
    frames, total, steps = [env.render()], 0.0, 0
    while steps < max_steps:
        if model is None:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action, _, _ = model.act(torch.from_numpy(norm(obs[None])), greedy=True)
            action = action[0]
        obs, reward, terminated, truncated, _ = env.step(
            np.clip(action, env.action_space.low, env.action_space.high))
        total += float(reward)
        steps += 1
        frames.append(env.render())
        if terminated or truncated:
            break
    env.close()
    return frames, total, steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="Walker2d-v5")
    ap.add_argument("--run", default=None)
    ap.add_argument("--checkpoint", default=None,
                    help="which checkpoint to film; default is the one the evaluation chose")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--size", type=int, default=480, help="render size, not agent input")
    ap.add_argument("--width", type=int, default=420, help="panel width in the film")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--gif-width", type=int, default=340, dest="gif_width")
    ap.add_argument("--gif-frames", type=int, default=60, dest="gif_frames")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run or HERE / "runs" / args.env.split("-")[0].lower())
    untrained_path = run / "ckpt_000.pt"
    # Film the checkpoint the evaluation chose, not the last one written. This
    # run peaked at 700k steps and lost 858 return by 2M, so "the trained agent"
    # and "the final file on disk" are two different policies.
    trained_path = run / "policy.pt"
    chosen = "final checkpoint"
    eval_path = run / "eval.json"
    if args.checkpoint:
        trained_path, chosen = run / args.checkpoint, args.checkpoint
    elif eval_path.exists():
        import json
        best = json.loads(eval_path.read_text())["best"]["checkpoint"]
        trained_path, chosen = run / best, f"best checkpoint ({best})"
    print(f"  filming {chosen}")

    strips = []
    for label, path in (("untrained", untrained_path), ("trained (PPO)", trained_path)):
        model, norm, state = load(path)
        frames, total, steps = record(model, norm, args.env, args.seed, args.size)
        print(f"  {label:<14} return {total:>8.1f} over {steps} steps "
              f"({state['step']:,} training steps)", flush=True)
        strips.append((label, f"{args.env} · return {total:.0f} · {steps} steps", frames))

    out = Path(args.out or ROOT / "docs" / "media" / "walker"
               / f"{args.env.split('-')[0].lower()}.mp4")
    frames = side_by_side(strips, width=args.width)
    print(f"\n{len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}")
    write_clip(frames, out, fps=args.fps, gif=True,
               gif_width=args.gif_width, gif_max_frames=args.gif_frames)


if __name__ == "__main__":
    main()
