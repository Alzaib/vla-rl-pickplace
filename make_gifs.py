"""
make_gifs.py -- render demo GIFs (with captions) of the expert, the BC model and the RL model acting in the simulator.

    python make_gifs.py                                              # all six GIFs into gifs_demo/
    python make_gifs.py --bc-ckpt g0_chunk.pt --rl-ckpt vla_grpo_chunk_best.pt
    python make_gifs.py --only expert bc                             # just some of them
    python make_gifs.py --bc-mode sampled                            # BC with sampled instead of argmax actions

Output files (gifs_demo/):
    1_expert.gif                    scripted expert (always succeeds)
    2_bc_success.gif                BC model, first scene it solves
    3_rl_sampled_success.gif        BC+RL model, SAMPLED actions, first scene it solves
    3_rl_sampled_fail.gif           BC+RL model, SAMPLED actions, first scene it fails
    4_rl_argmax_success.gif         BC+RL model, ARGMAX actions, first scene it solves
    4_rl_argmax_fail.gif            BC+RL model, ARGMAX actions, first scene it fails

SCENES (--scene-mode):
    distinct (default)  every GIF gets its OWN scene: scenes are tried in order from --seed0 and a scene already used by an
                        earlier GIF is skipped.  (Without this, all the "success" GIFs would land on the very first scene,
                        because most policies solve it.)
    first               the first scene (from --seed0) with the wanted outcome; successes then usually share one scene.
    --scene N           use scene N for every policy, whatever the outcome (files are named ..._success / ..._fail).
                        Best for a side-by-side comparison of the policies on the same scene.
Runs are reproducible: same checkpoint + same arguments = same GIFs. Every frame has a caption: who is acting, scene seed, instruction, step, action and the final result.
The model acts with "replan every step" (only chunk step 0 is executed), exactly like in evaluation and RL.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import chunk_data as C
import pickplace_env as E

SCALE, BAND = 8, 92                                     # pixels per image pixel, caption band height
GRIP = {0: "open", 1: "CLOSE"}
GREEN, RED, DARK, GREY = (20, 130, 50), (200, 30, 30), (20, 20, 20), (90, 90, 90)


def _font(size):
    import matplotlib
    path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def compose(frame, lines):
    """frame uint8 (72,84,3) -> PIL image with a white caption band on top. lines = [(text, colour, size), ...]"""
    pic = Image.fromarray(np.kron(frame, np.ones((SCALE, SCALE, 1), np.uint8)))
    out = Image.new("RGB", (pic.width, pic.height + BAND), (255, 255, 255))
    out.paste(pic, (0, BAND))
    d, y = ImageDraw.Draw(out), 5
    for text, col, size in lines:
        d.text((8, y), text, fill=col, font=_font(size))
        y += size + 6
    return out


def _caption(who, seed, instr, step, act=None, result=None):
    l3 = f"step {step}" if act is None else f"step {step}   move ({act[0]:+d},{act[1]:+d})   gripper {GRIP[act[2]]}"
    lines = [(who, DARK, 17), (f'scene {seed}: "{instr}"', GREY, 14), (l3, DARK, 14)]
    if result is not None:
        lines[2] = (l3 + "    " + ("SUCCESS" if result else "FAILED (60-step limit)"), GREEN if result else RED, 15)
    return lines


# ----------------------------------------------------------------------------- episode runners
def expert_episode(seed, who="Scripted expert"):
    env = E.PickPlaceEnv("sparse")
    obs = env.reset(seed=seed)
    instr = obs["instruction"]
    frames, done, step, info = [compose(env.render(), _caption(who, seed, instr, 0))], False, 0, {}
    while not done:
        a = tuple(int(v) for v in env.expert_action())
        _, _, done, info = env.step(a)
        step += 1
        frames.append(compose(env.render(), _caption(who, seed, instr, step, a, bool(info["success"]) if done else None)))
    return bool(info["success"]), step, frames


def model_episode(model, device, seed, greedy, who):
    from watch_policy import choose                      # imported here so the expert GIF works without torch
    env = E.PickPlaceEnv("sparse")
    obs = env.reset(seed=seed)
    instr, H = obs["instruction"], model.H
    rng = np.random.default_rng(seed)                    # sampled actions are reproducible per scene
    hist = np.tile(C.PAD, (H, 1)).astype(np.int64)
    frames, done, step, info = [compose(env.render(), _caption(who, seed, instr, 0))], False, 0, {}
    while not done:
        cls = choose(model, device, obs, hist, rng, greedy)
        a, ee0 = (cls[0] - 1, cls[1] - 1, cls[2]), env.ee
        obs, _, done, info = env.step(a)
        if H > 0:
            eff = [int(np.clip(env.ee[0] - ee0[0], -1, 1)) + 1, int(np.clip(env.ee[1] - ee0[1], -1, 1)) + 1, cls[2]]
            hist = np.concatenate([hist[1:], np.array([eff])])
        step += 1
        frames.append(compose(env.render(), _caption(who, seed, instr, step, a, bool(info["success"]) if done else None)))
    return bool(info["success"]), step, frames


def find(run, want_success, seed0, max_tries, skip=()):
    """first scene (from seed0 on, not in `skip`) whose outcome equals want_success"""
    for seed in range(seed0, seed0 + max_tries):
        if seed in skip:
            continue
        ok, steps, frames = run(seed)
        if ok == want_success:
            return seed, steps, frames
    return None


def save(frames, path, ms, hold=12):
    frames[0].save(path, save_all=True, append_images=frames[1:] + [frames[-1]] * hold, duration=ms, loop=0)


# ----------------------------------------------------------------------------- main
def main(a):
    want = set(a.only)
    os.makedirs(a.out, exist_ok=True)
    jobs = []                                            # (file name, runner, want_success)
    if "expert" in want:
        jobs.append(("1_expert.gif", lambda s: expert_episode(s), True))
    device = bc = rl = None
    if want & {"bc", "rl_sampled", "rl_argmax"}:
        from train_chunk import load_ckpt
        from train_vla import get_device
        device = get_device(a.device)
        load = lambda p: load_ckpt(p, device)[0].eval()
        if "bc" in want:
            bc = load(a.bc_ckpt)
            g = a.bc_mode == "greedy"
            jobs.append(("2_bc_success.gif", lambda s: model_episode(bc, device, s, g, f"BC model ({'argmax' if g else 'sampled'} actions)"), True))
        if want & {"rl_sampled", "rl_argmax"}:
            rl = load(a.rl_ckpt)
            if "rl_sampled" in want:
                f = lambda s: model_episode(rl, device, s, False, "BC + RL model (sampled actions)")
                jobs += [("3_rl_sampled_success.gif", f, True), ("3_rl_sampled_fail.gif", f, False)]
            if "rl_argmax" in want:
                f = lambda s: model_episode(rl, device, s, True, "BC + RL model (argmax actions)")
                jobs += [("4_rl_argmax_success.gif", f, True), ("4_rl_argmax_fail.gif", f, False)]
    used = set()
    if a.scene is not None:                              # same scene for every policy, outcome decided by the policy
        done = set()
        for name, run, _ in jobs:
            stem = name.split("_success")[0].split("_fail")[0].replace(".gif", "")
            if stem in done:                             # the success/fail jobs share one runner -> run it once
                continue
            done.add(stem)
            ok, steps, frames = run(a.scene)
            fn = f"{stem}_{'success' if ok else 'fail'}.gif" if stem != "1_expert" else "1_expert.gif"
            save(frames, os.path.join(a.out, fn), a.ms)
            print(f"{fn:28s} scene {a.scene}  {steps:2d} steps  ({'SUCCESS' if ok else 'FAILED'})")
    else:
        for name, run, want_success in jobs:
            r = find(run, want_success, a.seed0, a.max_tries, used if a.scene_mode == "distinct" else ())
            if r is None:
                print(f"{name:28s} no {'successful' if want_success else 'failed'} episode in {a.max_tries} scenes -> not written")
                continue
            seed, steps, frames = r
            used.add(seed)
            save(frames, os.path.join(a.out, name), a.ms)
            print(f"{name:28s} scene {seed}  {steps:2d} steps  ({len(frames)} frames)")
    print(f"\nGIFs are in {a.out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bc-ckpt", default="g0_chunk.pt")
    ap.add_argument("--rl-ckpt", default="vla_grpo_chunk_best.pt")
    ap.add_argument("--bc-mode", choices=["greedy", "sampled"], default="greedy")
    ap.add_argument("--only", nargs="+", choices=["expert", "bc", "rl_sampled", "rl_argmax"],
                    default=["expert", "bc", "rl_sampled", "rl_argmax"])
    ap.add_argument("--seed0", type=int, default=2_000_000)
    ap.add_argument("--scene-mode", choices=["distinct", "first"], default="distinct",
                    help="distinct: every GIF gets its own scene (default); first: first scene with the wanted outcome")
    ap.add_argument("--scene", type=int, default=None, help="use this one scene for every policy (outcome not forced)")
    ap.add_argument("--max-tries", type=int, default=300)
    ap.add_argument("--ms", type=int, default=150, help="milliseconds per frame")
    ap.add_argument("--out", default="gifs_demo")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    main(ap.parse_args())