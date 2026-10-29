"""
make_dataset.py -- collect, load and inspect a demonstration dataset for a small VLA.

  python make_dataset.py collect --train 1000 --val 100 --test 100 --out vla_data.npz
  python make_dataset.py inspect --data vla_data.npz       # summary + episode_preview.png
  python make_dataset.py baselines                         # sanity-check the test harness
  python make_dataset.py merge a.npz b.npz --out all.npz   # combine e.g. scripted + human data

Every ROW of the dataset is one timestep t of one episode:

  image            (N, H, W, 3) uint8   what the camera sees at time t       <- VLA input
  instruction      per-episode string   the task description                 <- VLA input
  t, time_s        (N,)                 step index and t * DT seconds        <- VLA input (optional)
  action           (N, 3) int8          expert action (dx, dy, grip)         <- VLA TARGET
  ee_pos, gripper_closed, holding, block_pos   ground-truth STATE at time t  <- supervision / debugging
  reward, done     (N,)                 what the env returned AFTER the action (for the RL stage)
  executed_expert  (N,) bool            True if the executed action == the stored `action`; False if a random
                                        action was executed instead (see --noise). Always True for human rows.

Per-EPISODE arrays (E,): instruction, success, length, split (0 train / 1 val / 2 test), seed,
target_block, target_region, block_color, region_color, obstacle.
Rows of episode e are  episode_start[e] : episode_start[e] + episode_len[e].

About --noise: with probability `noise` per step the *executed* action is random, but the stored
`action` is still what the expert would do in that state. The data then contains "recovery"
situations, which makes a cloned policy far more robust. Use --noise 0 for clean demos.
"""
import argparse
import json

import numpy as np

import pickplace_env as E
from pickplace_env import PickPlaceEnv

SPLITS = {"train": 0, "val": 1, "test": 2}
# test scenes (seed >= 2,000,000) are used by E.evaluate; train/val/test dataset seeds never overlap them
SEED_BASE = {"train": 0, "val": 1_000_000, "test": 1_500_000}


# ============================================================================ writing
STEP_KEYS = ["image", "t", "time_s", "action", "ee_pos", "gripper_closed", "holding", "block_pos",
             "reward", "done", "executed_expert"]


class DatasetWriter:
    """Collects episodes (scripted OR human) and turns them into the flat arrays described above."""

    def __init__(self, px, noise=None, source="expert"):
        self.px, self.noise, self.source = px, noise, source
        self.S = {k: [] for k in STEP_KEYS + ["episode_id"]}
        self.EP = {k: [] for k in ["instruction", "success", "length", "split", "seed", "target_block", "target_region",
                                   "block_color", "region_color", "obstacle", "episode_start", "source"]}
        self.rows = 0

    def add_episode(self, rows, instruction, success, split, seed, st0, source=None):
        """rows: list of per-step dicts with the STEP_KEYS; st0: env.state() at the start of the episode."""
        ep_id = len(self.EP["length"])
        for r in rows:
            for k in STEP_KEYS:
                self.S[k].append(r[k])
            self.S["episode_id"].append(ep_id)
        for k, v in dict(instruction=instruction, success=success, length=len(rows), split=SPLITS[split], seed=seed,
                         target_block=st0["target_block"], target_region=st0["target_region"],
                         block_color=st0["block_color"], region_color=st0["region_color"],
                         obstacle=st0["obstacle"], episode_start=self.rows, source=source or self.source).items():
            self.EP[k].append(v)
        self.rows += len(rows)

    def n_episodes(self):
        return len(self.EP["length"])

    def to_arrays(self):
        A = {k: np.array(v) for k, v in self.S.items()}
        A["image"] = A["image"].astype(np.uint8); A["action"] = A["action"].astype(np.int8)
        A["t"] = A["t"].astype(np.int16); A["time_s"] = A["time_s"].astype(np.float32)
        A["reward"] = A["reward"].astype(np.float32)
        for k, v in self.EP.items():
            A["ep_" + k] = np.array(v)
        A["ep_length"] = A["ep_length"].astype(np.int16)
        meta = dict(grid=E.G, margin_cells=E.MARGIN, px=self.px, dt=E.DT, max_steps=E.MAX_STEPS, colors=E.COLORS,
                    noise=self.noise, action="(dx, dy, grip): dx,dy in {-1,0,1}; grip 0=open 1=close",
                    image="uint8 (H, W, 3); row 0 = top = largest y; robot base on the left")
        A["meta"] = np.array(json.dumps(meta))
        return A


def load_arrays(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def merge_arrays(a, b):
    """Concatenate two datasets (same image size). Episode ids / start rows of b are shifted."""
    assert a["image"].shape[1:] == b["image"].shape[1:], "image sizes differ (different --px?)"
    n_ep, n_rows = len(a["ep_length"]), len(a["t"])
    out = {}
    for k in a:
        if k == "meta":
            out[k] = a[k]
            continue
        bv = b[k]
        if k == "episode_id":
            bv = bv + n_ep
        elif k == "ep_episode_start":
            bv = bv + n_rows
        out[k] = np.concatenate([a[k], bv])
    return out


def save_arrays(arrays, path):
    np.savez_compressed(path, **arrays)


# ============================================================================ scripted collection
def collect(n_train, n_val, n_test, px, noise, out, seed=0):
    rng = np.random.default_rng(seed + 31337)
    env = PickPlaceEnv("shaped", px=px)
    W = DatasetWriter(px, noise, "expert")
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        for i in range(n):
            sd = SEED_BASE[split] + seed * 100_000 + i
            obs = env.reset(seed=sd)
            st0, rows = env.state(), []
            while True:
                st = env.state()
                a_exp = env.expert_action()
                use_expert = rng.random() >= noise
                a_exec = a_exp if use_expert else (int(rng.integers(-1, 2)), int(rng.integers(-1, 2)), int(rng.integers(2)))
                row = dict(image=obs["image"], t=obs["t"], time_s=obs["time_s"], action=a_exp, ee_pos=st["ee"],
                           gripper_closed=st["gripper_closed"], holding=st["holding"], block_pos=st["block_pos"],
                           executed_expert=use_expert)
                obs, row["reward"], row["done"], info = env.step(a_exec)
                rows.append(row)
                if row["done"]:
                    break
            W.add_episode(rows, env.instruction, info["success"], split, sd, st0)
        print(f"  collected {n} {split} episodes")
    save_arrays(W.to_arrays(), out)
    print(f"saved {W.rows} timesteps / {W.n_episodes()} episodes -> {out}")


def merge(paths, out):
    arrays = load_arrays(paths[0])
    for p in paths[1:]:
        arrays = merge_arrays(arrays, load_arrays(p))
    save_arrays(arrays, out)
    print(f"merged {len(paths)} files -> {out}: {len(arrays['t'])} timesteps, {len(arrays['ep_length'])} episodes")


# ============================================================================ loading
class VLADataset:
    """
        ds = VLADataset("vla_data.npz", split="train")
        ex = ds[0]   -> dict(image, instruction, t, time_s, action, ee_pos, gripper_closed, holding, block_pos, ...)
        for ep in ds.episodes(): ...       # one whole demonstration at a time
        for batch in ds.batches(64): ...   # dict of stacked arrays (instruction is a list of str)
    """

    def __init__(self, path, split=None):
        d = np.load(path, allow_pickle=False)
        self.d = {k: d[k] for k in d.files}
        self.meta = json.loads(str(self.d["meta"]))
        ep_split = self.d["ep_split"]
        self.ep_ids = np.nonzero(ep_split == SPLITS[split])[0] if split else np.arange(len(ep_split))
        sel = np.isin(self.d["episode_id"], self.ep_ids)
        self.idx = np.nonzero(sel)[0]

    def __len__(self):
        return len(self.idx)

    def _row(self, i):
        d, e = self.d, self.d["episode_id"][i]
        out = {k: d[k][i] for k in ["image", "t", "time_s", "action", "ee_pos", "gripper_closed", "holding",
                                    "block_pos", "reward", "done", "executed_expert"]}
        out["instruction"] = str(d["ep_instruction"][e]); out["episode_id"] = int(e)
        return out

    def __getitem__(self, k):
        return self._row(self.idx[k])

    def episodes(self):
        for e in self.ep_ids:
            s, n = self.d["ep_episode_start"][e], self.d["ep_length"][e]
            rows = np.arange(s, s + n)
            ep = {k: self.d[k][rows] for k in ["image", "t", "time_s", "action", "ee_pos", "gripper_closed",
                                               "holding", "block_pos", "reward", "done"]}
            ep.update(instruction=str(self.d["ep_instruction"][e]), success=bool(self.d["ep_success"][e]), episode_id=int(e))
            yield ep

    def batches(self, batch_size, shuffle=True, seed=0):
        order = np.random.default_rng(seed).permutation(len(self.idx)) if shuffle else np.arange(len(self.idx))
        for i in range(0, len(order), batch_size):
            rows = self.idx[order[i:i + batch_size]]
            b = {k: self.d[k][rows] for k in ["image", "t", "time_s", "action", "ee_pos", "gripper_closed",
                                              "holding", "block_pos", "reward", "done"]}
            b["instruction"] = [str(self.d["ep_instruction"][e]) for e in self.d["episode_id"][rows]]
            yield b


# ============================================================================ inspection
def inspect(path, preview="episode_preview.png"):
    ds = VLADataset(path)
    d, m = ds.d, ds.meta
    print(f"file: {path}\nmeta: {json.dumps(m)}\n")
    print(f"timesteps: {len(d['t'])}   episodes: {len(d['ep_length'])}   image shape: {d['image'].shape[1:]}")
    for name, v in SPLITS.items():
        sel = d["ep_split"] == v
        print(f"  {name:5s}: {int(sel.sum()):5d} episodes, {int(d['ep_length'][sel].sum()):6d} steps, "
              f"expert success {d['ep_success'][sel].mean() if sel.any() else float('nan'):.2f}, "
              f"mean length {d['ep_length'][sel].mean() if sel.any() else float('nan'):.1f}")
    a = d["action"]
    if "ep_source" in d:
        src, cnt = np.unique(d["ep_source"], return_counts=True)
        print("  episodes by source:", dict(zip(src.tolist(), cnt.tolist())))
    print("\naction histogram:")
    print("  dx   :", {int(k): int((a[:, 0] == k).sum()) for k in (-1, 0, 1)})
    print("  dy   :", {int(k): int((a[:, 1] == k).sum()) for k in (-1, 0, 1)})
    print("  grip :", {"open": int((a[:, 2] == 0).sum()), "close": int((a[:, 2] == 1).sum())})
    print(f"  steps where the executed action was a random one: {(~d['executed_expert']).mean():.1%}")
    print("\ndistinct instructions:", len(set(d["ep_instruction"].tolist())))
    ep = next(ds.episodes())
    print(f"\nexample episode 0: \"{ep['instruction']}\"  ({len(ep['t'])} steps, success={ep['success']})")
    print("   t  time_s  action(dx,dy,grip)  ee_pos  gripper  holding  reward")
    for i in range(len(ep["t"])):
        print(f"  {ep['t'][i]:2d}  {ep['time_s'][i]:5.1f}   {tuple(int(v) for v in ep['action'][i])!s:14s}   "
              f"{tuple(int(v) for v in ep['ee_pos'][i])!s:8s} {'closed' if ep['gripper_closed'][i] else 'open':7s} "
              f"{int(ep['holding'][i]):4d}   {ep['reward'][i]:+.2f}")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = min(len(ep["t"]), 8)
        idx = np.unique(np.linspace(0, len(ep["t"]) - 1, n).astype(int))
        fig, axs = plt.subplots(1, len(idx), figsize=(2.6 * len(idx), 2.9))
        for ax, i in zip(np.atleast_1d(axs), idx):
            ax.imshow(ep["image"][i]); ax.axis("off")
            dx, dy, g = (int(v) for v in ep["action"][i])
            ax.set_title(f"t={ep['t'][i]}\n({dx:+d},{dy:+d},{'close' if g else 'open'})", fontsize=8)
        fig.suptitle(f'"{ep["instruction"]}"   title = ground-truth action', fontsize=10)
        fig.savefig(preview, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"\nsaved {preview}")
    except ImportError:
        print("\n(matplotlib not installed: skipped preview image)")


# ============================================================================ sanity baselines
def baselines():
    rng = np.random.default_rng(0)
    rand = lambda img, ins, t: (int(rng.integers(-1, 2)), int(rng.integers(-1, 2)), int(rng.integers(2)))
    print("random policy :", E.evaluate(rand, 50))
    # the expert needs the env itself, so we run it directly
    env, ok = PickPlaceEnv(), 0
    for i in range(50):
        env.reset(seed=2_000_000 + i)
        while True:
            _, _, done, info = env.step(env.expert_action())
            if done:
                break
        ok += info["success"]
    print("scripted expert: success_rate", ok / 50, " (upper bound for imitation learning)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["collect", "inspect", "baselines", "merge"])
    ap.add_argument("--train", type=int, default=1000)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--px", type=int, default=6, help="pixels per grid cell (image is 12*px high)")
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="vla_data.npz")
    ap.add_argument("--data", default="vla_data.npz")
    ap.add_argument("files", nargs="*", help="for `merge`: input .npz files")
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.train, a.val, a.test, a.px, a.noise, a.out, a.seed)
    elif a.cmd == "inspect":
        inspect(a.data)
    elif a.cmd == "merge":
        merge(a.files, a.out)
    else:
        baselines()
