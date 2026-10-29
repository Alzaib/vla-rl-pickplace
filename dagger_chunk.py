"""
dagger_chunk.py -- DAgger for the action-chunking VLA.

    python dagger_chunk.py --policy c.pt --episodes 300 --beta 0.3 --out dagger_c1.npz
    python train_chunk.py --extra dagger_c1.npz --best-path c_dag1.pt          # chunk targets are rebuilt automatically

The CHUNK MODEL drives the simulator (it re-plans every step and executes its first action, sampled), with its
own performed actions fed back as history.  Every visited state is stored with the SCRIPTED EXPERT's action as
the label (and train_chunk.py later rebuilds the expert's K-step plan from that exact state).

Why this is the right fix for the history shortcut: in expert demonstrations the previous action almost
determines the next one ("keep going the same way"), so a model can score well offline by copying its history.
In closed loop its history comes from ITS OWN behaviour, so the shortcut points the wrong way.  Here the history
is the learner's, the label is the expert's, so copying the history is punished.

  --beta  probability that the expert's action is executed instead of the learner's (0.3 first round, then 0.15, 0).
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

import chunk_data as C
import pickplace_env as E
from make_dataset import DatasetWriter, save_arrays
from train_chunk import load_ckpt
from train_vla import get_device
from vla_data import tokenize


@torch.no_grad()
def collect(model, device, episodes, beta=0.3, seed0=8_000_000, parallel=32, rng_seed=0, greedy=False):
    rng = np.random.default_rng(rng_seed)
    H = model.H
    model.eval()
    W = DatasetWriter(6, noise=None, source="dagger")
    succ_all, len_all, done_eps = [], [], 0
    while done_eps < episodes:
        n = min(parallel, episodes - done_eps)
        seeds = [seed0 + done_eps + i for i in range(n)]
        envs = [E.PickPlaceEnv("shaped") for _ in range(n)]
        obs = [e.reset(seed=s) for e, s in zip(envs, seeds)]
        st0 = [e.state() for e in envs]
        toks = np.array([tokenize(o["instruction"]) for o in obs], np.int64)
        hist = np.tile(C.PAD, (n, H, 1))
        rows, succ, active = [[] for _ in range(n)], [False] * n, list(range(n))
        while active:
            img = torch.from_numpy(np.stack([obs[i]["image"] for i in active])).to(device).permute(0, 3, 1, 2).float() / 255.0
            tt = torch.tensor([obs[i]["t"] / E.MAX_STEPS for i in active], dtype=torch.float32, device=device)
            hh = torch.from_numpy(hist[active]).to(device) if H > 0 else None
            out = model(img, torch.from_numpy(toks[active]).to(device), tt, hh)
            pr = [F.softmax(out[k][:, 0], -1).cpu().numpy() for k in ("dx", "dy", "grip")]     # first chunk step only
            nxt = []
            for j, i in enumerate(active):
                env = envs[i]
                cls = [int(p[j].argmax()) if greedy else int(rng.choice(len(p[j]), p=p[j] / p[j].sum())) for p in pr]
                learner = (cls[0] - 1, cls[1] - 1, cls[2])
                label = env.expert_action()                          # <- training target (state-dependent)
                st = env.state()
                exec_a = label if rng.random() < beta else learner
                row = dict(image=obs[i]["image"], t=obs[i]["t"], time_s=obs[i]["time_s"], action=label,
                           ee_pos=st["ee"], gripper_closed=st["gripper_closed"], holding=st["holding"],
                           block_pos=st["block_pos"], executed_expert=(tuple(exec_a) == tuple(label)))
                ee0 = env.ee
                obs[i], row["reward"], row["done"], info = env.step(exec_a)
                rows[i].append(row)
                if H > 0:                                            # same definition as chunk_data.effective_history
                    eff = [int(np.clip(env.ee[0] - ee0[0], -1, 1)) + 1, int(np.clip(env.ee[1] - ee0[1], -1, 1)) + 1, int(exec_a[2])]
                    hist[i] = np.concatenate([hist[i][1:], np.array([eff])])
                if row["done"]:
                    succ[i] = info["success"]
                else:
                    nxt.append(i)
            active = nxt
        for i in range(n):
            W.add_episode(rows[i], envs[i].instruction, succ[i], "train", seeds[i], st0[i], source="dagger")
            succ_all.append(succ[i]); len_all.append(len(rows[i]))
        done_eps += n
    return W, dict(episodes=episodes, rows=W.rows, success_rate=float(np.mean(succ_all)), mean_len=float(np.mean(len_all)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--seed0", type=int, default=8_000_000, help="use a new range every round")
    ap.add_argument("--parallel", type=int, default=32)
    ap.add_argument("--greedy", action="store_true", help="learner uses argmax instead of sampling")
    ap.add_argument("--out", default="dagger_c1.npz")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    a = ap.parse_args()
    dev = get_device(a.device)
    model, _ = load_ckpt(a.policy, dev)
    t0 = time.time()
    W, st = collect(model, dev, a.episodes, a.beta, a.seed0, a.parallel, greedy=a.greedy)
    save_arrays(W.to_arrays(), a.out)
    print(f"saved {st['rows']} labelled states from {st['episodes']} episodes -> {a.out}  [{time.time() - t0:.0f}s]")
    print(f"learner success while collecting (beta={a.beta}): {st['success_rate']:.2f}   mean length {st['mean_len']:.1f}")
