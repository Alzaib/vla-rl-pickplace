"""
vla_data.py -- turns the .npz datasets into plain arrays a VLA can train on.  (NumPy only, no torch)

  tokenize("put the red block in the blue container") -> 8 integer ids
  load_expert(path, "train")        -> arrays for one split of a dataset
  load_human(path, n_holdout)       -> (train_arrays, holdout_arrays), split by EPISODE
  concat(a, b)                      -> stack two array dicts

An "arrays" dict has:
  image   uint8   (N, 72, 84, 3)
  tokens  int64   (N, 8)       instruction word ids
  t       float32 (N,)         step index (divide by MAX_STEPS yourself)
  action  int64   (N, 3)       CLASS indices: dx+1, dy+1, grip     (dx,dy in {-1,0,1} -> {0,1,2})
  aux     int64   (N, 6)       auxiliary "where is everything" labels, each a cell coordinate 0..11:
                               [ee_x, ee_y, target_block_x, target_block_y, target_region_x, target_region_y]
                               (taken from the stored ground-truth state; used as an extra training signal only)
"""
import numpy as np

import pickplace_env as E
from make_dataset import SPLITS, load_arrays

VOCAB = ["<pad>", "put", "the"] + E.COLORS + ["block", "in", "on", "container", "zone"]
WORD2ID = {w: i for i, w in enumerate(VOCAB)}
MAX_TEXT_LEN = 8
ACTION_SHIFT = np.array([1, 1, 0], dtype=np.int64)       # (dx,dy,grip) -> class ids


def tokenize(text):
    ids = [WORD2ID[w] for w in text.lower().split()]
    assert len(ids) <= MAX_TEXT_LEN, f"instruction too long: {text}"
    return ids + [0] * (MAX_TEXT_LEN - len(ids))


def decode_action(dx_cls, dy_cls, grip_cls):
    """class ids back to env action (dx, dy, grip)."""
    return int(dx_cls) - 1, int(dy_cls) - 1, int(grip_cls)


AUX_NAMES = ["ee_x", "ee_y", "target_block_x", "target_block_y", "target_region_x", "target_region_y"]


def _region_centers(d):
    """Target-region centre per episode. It isn't stored per step, but scenes are reproducible from
    their seed, so we rebuild each scene (fast) and read it off."""
    env = E.PickPlaceEnv()
    out = np.zeros((len(d["ep_seed"]), 2), np.int64)
    for e, (sd, tr) in enumerate(zip(d["ep_seed"], d["ep_target_region"])):
        env.reset(seed=int(sd))
        assert env.instruction == str(d["ep_instruction"][e]), "scene regeneration mismatch"
        r = env.regions[int(tr)]
        out[e] = (r["cx"], r["cy"])
    return out


def _arrays(d, rows):
    table = np.array([tokenize(s) for s in d["ep_instruction"]], dtype=np.int64)     # (episodes, 8)
    ep = d["episode_id"][rows]
    actions = d["action"][rows].astype(np.int64) + ACTION_SHIFT
    # prev 4 actions
    prev_actions = np.zeros((len(rows), 4, 3), dtype=np.int64)

    row_to_ep = ep
    for i in range(len(rows)):
        for k in range(1, 5):
            j = i - k
            if j >= 0 and row_to_ep[j] == row_to_ep[i]:
                prev_actions[i, 4 - k] = actions[j]
    centers = _region_centers(d)
    tb_pos = d["block_pos"][rows, d["ep_target_block"][ep]]                           # (N, 2)
    aux = np.concatenate([d["ee_pos"][rows], tb_pos, centers[ep]], 1).astype(np.int64)
    return dict(image=d["image"][rows],
                tokens=table[ep],
                t=d["t"][rows].astype(np.float32),
                action=actions,
                aux=aux,
                prev_actions=prev_actions)


def _rows_of(d, episode_ids):
    return np.nonzero(np.isin(d["episode_id"], episode_ids))[0]


def load_expert(path, split="train", max_episodes=None):
    d = load_arrays(path)
    eps = np.nonzero(d["ep_split"] == SPLITS[split])[0]
    if max_episodes:
        eps = eps[:max_episodes]
    return _arrays(d, _rows_of(d, eps))


def load_human(path, n_holdout=5):
    """All episodes of a human file; the LAST n_holdout episodes are kept out for testing."""
    d = load_arrays(path)
    eps = np.arange(len(d["ep_length"]))
    n_holdout = min(n_holdout, max(len(eps) - 1, 0))
    train_eps, hold_eps = eps[:len(eps) - n_holdout], eps[len(eps) - n_holdout:]
    hold = _arrays(d, _rows_of(d, hold_eps)) if n_holdout else None
    return _arrays(d, _rows_of(d, train_eps)), hold


def concat(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def n_rows(arrays):
    return len(arrays["t"])