"""
watch_policy.py -- WATCH a trained chunk-VLA act in the simulator (live window and/or GIF files).

    python watch_policy.py --ckpt vla_grpo_chunk_best.pt                     # live window, 5 episodes, sampled actions
    python watch_policy.py --ckpt g0_chunk.pt --mode greedy                  # same for the BC model, argmax actions
    python watch_policy.py --ckpt vla_grpo_chunk_best.pt --only-fail         # show only episodes that fail
    python watch_policy.py --ckpt vla_grpo_chunk_best.pt --gif-dir gifs --no-window --episodes 20   # save GIFs, no window

The model acts with "replan every step" (execute=1), the same setting that was evaluated and used in RL.
Episode seeds start at --seed0 (default 2_000_000 = the evaluation scenes), so scene N here is scene N in the evaluation.
Keys in the window: close the window to stop.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

import pickplace_env as E
import chunk_data as C
from train_chunk import load_ckpt
from train_vla import get_device
from vla_data import tokenize

GRIP = {0: "open", 1: "CLOSE"}


@torch.no_grad()
def choose(model, device, obs, hist, rng, greedy):
    img = torch.from_numpy(obs["image"][None]).to(device).permute(0, 3, 1, 2).float() / 255.0
    tok = torch.from_numpy(np.array(tokenize(obs["instruction"]), np.int64)[None]).to(device)
    tt = torch.tensor([obs["t"] / E.MAX_STEPS], dtype=torch.float32, device=device)
    hh = torch.from_numpy(hist[None]).to(device) if model.H > 0 else None
    out = model(img, tok, tt, hh)
    cls = []
    for k in ("dx", "dy", "grip"):
        p = F.softmax(out[k][0, 0], -1).cpu().numpy()                 # chunk step 0 = the action that is executed
        cls.append(int(p.argmax()) if greedy else int(rng.choice(len(p), p=p / p.sum())))
    return cls


def up(img, s=10):
    return np.kron(img, np.ones((s, s, 1), np.uint8))


def run_episode(model, device, seed, greedy, rng, show=None, pause=0.12):
    env = E.PickPlaceEnv("sparse")
    obs = env.reset(seed=seed)
    H = model.H
    hist = np.tile(C.PAD, (H, 1)).astype(np.int64)
    frames, info, done, step = [env.render()], {}, False, 0
    instr = obs["instruction"]
    if show:
        show(env.render(), f"seed {seed}: {instr}\nstep 0", pause)
    while not done:
        cls = choose(model, device, obs, hist, rng, greedy)
        ee0 = env.ee
        obs, r, done, info = env.step((cls[0] - 1, cls[1] - 1, cls[2]))
        if H > 0:
            eff = [int(np.clip(env.ee[0] - ee0[0], -1, 1)) + 1, int(np.clip(env.ee[1] - ee0[1], -1, 1)) + 1, cls[2]]
            hist = np.concatenate([hist[1:], np.array([eff])])
        step += 1
        frames.append(env.render())
        bump = "  (bumped)" if info.get("collided") else ""
        title = f"seed {seed}: {instr}\nstep {step}  move ({cls[0] - 1:+d},{cls[1] - 1:+d})  gripper {GRIP[cls[2]]}{bump}"
        if done:
            title += "\n>>> SUCCESS" if info["success"] else "\n>>> FAILED (time out)"
        if show:
            show(frames[-1], title, pause * (6 if done else 1))
    return bool(info.get("success", False)), step, frames, instr


def save_gif(frames, path, ms=120):
    from PIL import Image
    ims = [Image.fromarray(up(f)) for f in frames] + [Image.fromarray(up(frames[-1]))] * 8   # hold the last frame
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=ms, loop=0)


def main(a):
    device = get_device(a.device)
    model, cfg = load_ckpt(a.ckpt, device)
    model.eval()
    rng = np.random.default_rng(a.seed)
    show = None
    if not a.no_window:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        im = ax.imshow(up(np.zeros((72, 84, 3), np.uint8)), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])

        def show(frame, title, pause):
            if not plt.fignum_exists(fig.number):
                raise SystemExit
            im.set_data(up(frame)); ax.set_title(title, fontsize=11)
            fig.canvas.draw_idle(); plt.pause(pause)
    if a.gif_dir:
        os.makedirs(a.gif_dir, exist_ok=True)
    n_ok = n_run = 0
    seed = a.seed0
    shown = 0
    while shown < a.episodes and n_run < a.max_tries:
        # only-fail: run silently first; replay only if it failed (same seed + same rng state = same episode)
        state = rng.bit_generator.state
        ok, steps, frames, instr = run_episode(model, device, seed, a.mode == "greedy", rng, show=None)
        n_run += 1; n_ok += ok
        if a.only_fail and ok:
            seed += 1
            continue
        if show:
            rng.bit_generator.state = state
            ok, steps, frames, instr = run_episode(model, device, seed, a.mode == "greedy", rng, show=show, pause=a.pause)
        print(f"seed {seed}: {'SUCCESS' if ok else 'FAIL   '} in {steps:2d} steps | {instr}")
        if a.gif_dir:
            safe_instr = (
                instr.replace(" ", "_")
                    .replace("/", "_")
            )
            save_gif(frames, os.path.join(a.gif_dir, f"seed{seed}_{safe_instr}_{'ok' if ok else 'fail'}.gif"))
        shown += 1; seed += 1
    print(f"\nsuccess {n_ok}/{n_run} = {n_ok / max(n_run, 1):.2f}  (mode: {a.mode}, checkpoint: {a.ckpt})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="vla_grpo_chunk_best.pt")
    ap.add_argument("--mode", choices=["sampled", "greedy"], default="sampled")
    ap.add_argument("--episodes", type=int, default=5, help="how many episodes to show")
    ap.add_argument("--seed0", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0, help="random seed for action sampling")
    ap.add_argument("--only-fail", action="store_true", help="skip successful episodes")
    ap.add_argument("--max-tries", type=int, default=500)
    ap.add_argument("--pause", type=float, default=0.12, help="seconds per frame in the window")
    ap.add_argument("--gif-dir", default=None, help="also save every shown episode as a GIF here")
    ap.add_argument("--no-window", action="store_true", help="no live window (use with --gif-dir)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    main(ap.parse_args())
