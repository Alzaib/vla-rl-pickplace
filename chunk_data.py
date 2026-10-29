"""
chunk_data.py -- build ACTION-CHUNK training targets and ACTION-HISTORY inputs.  (NumPy only, no torch)

For every stored row (episode e, step t) we need, on top of the usual image / tokens / t / action / aux:

  chunk (N, K, 3) int64  targets [a_t, a_t+1, ..., a_t+K-1] as class ids (dx+1, dy+1, grip)
  cmask (N, K)    bool   False where the target does not exist (episode already finished)
  hist  (N, H, 3) int64  the last H actions that were ACTUALLY performed, oldest first, as class ids.
                         Before the episode started, entries are PAD = (3, 3, 2).
  gs    (N,)      int64  gripper state in the image (0 open / 1 closed) -- only used for a diagnostic

TWO DESIGN DECISIONS THAT MATTER

1) Where do chunk targets come from?
   "rollout" (default for scripted-expert / DAgger rows): rebuild the exact simulator state of the row, then
        let the expert play K steps FROM THAT STATE.  This is the plan the expert would follow starting here.
        Why not just read a_t+1, a_t+2 from the file?  In noisy-expert data (--noise 0.1) and in DAgger data
        the NEXT stored state came from a RANDOM or LEARNER action, so the next stored label is the expert's
        answer to a different situation, not a continuation of the plan from state t.
   "logged"  (used for human rows): the human's real future actions are the supervision, so we read them.

2) What counts as a "past action"?
   The effective action: the end-effector displacement that really happened (a bump into a wall = (0,0))
   and the gripper state after the step.  It is derived from consecutive stored states, so it works for any
   existing dataset file, and at inference time we compute it the same way, so train and test agree.
"""
import os

import numpy as np

import pickplace_env as E
import vla_data as D

PAD = np.array([3, 3, 2], np.int64)          # "no action yet" id for (dx, dy, grip)


# ----------------------------------------------------------------------------- history
def effective_history(d, H):
    """(N, H, 3) class ids of the previous H performed actions for every row of the file."""
    N = len(d["t"])
    ep, ee, gc = d["episode_id"], d["ee_pos"].astype(np.int64), d["gripper_closed"].astype(np.int64)
    eff = np.zeros((N, 3), np.int64)                       # eff[i] = action performed AT row i
    has_next = np.zeros(N, bool)
    same = ep[1:] == ep[:-1]
    has_next[:-1] = same
    dxy = np.clip(ee[1:] - ee[:-1], -1, 1)
    eff[:-1, 0], eff[:-1, 1] = dxy[:, 0] + 1, dxy[:, 1] + 1
    eff[:-1, 2] = gc[1:]                                   # gripper after step == the grip command
    hist = np.tile(PAD, (N, H, 1))
    for k in range(1, H + 1):                              # slot H-k holds the action performed k steps ago
        src = np.arange(N) - k
        ok = (src >= 0) & (ep[np.maximum(src, 0)] == ep) & has_next[np.maximum(src, 0)]
        hist[ok, H - k] = eff[src[ok]]
    return hist


# ----------------------------------------------------------------------------- chunk targets
def logged_chunks(d, K):
    N = len(d["t"])
    ep, act = d["episode_id"], d["action"].astype(np.int64) + D.ACTION_SHIFT
    chunk, mask = np.zeros((N, K, 3), np.int64), np.zeros((N, K), bool)
    for k in range(K):
        src = np.arange(N) + k
        ok = (src < N) & (ep[np.minimum(src, N - 1)] == ep)
        chunk[ok, k], mask[ok, k] = act[src[ok]], True
    return chunk, mask


def rollout_chunks(d, rows, K):
    """Expert plan of K steps from the exact state stored in each row (see module docstring)."""
    env = E.PickPlaceEnv("sparse")
    env.render = lambda: None                              # we never need pixels here -> ~10x faster
    chunk, mask = np.zeros((len(rows), K, 3), np.int64), np.zeros((len(rows), K), bool)
    last = -1
    for j, i in enumerate(rows):
        e = int(d["episode_id"][i])
        if e != last:
            env.reset(seed=int(d["ep_seed"][e])); last = e
            assert env.instruction == str(d["ep_instruction"][e]), "scene regeneration mismatch"
        env.ee = tuple(int(v) for v in d["ee_pos"][i])
        env.gripper_closed = bool(d["gripper_closed"][i])
        h = int(d["holding"][i]); env.held = None if h < 0 else h
        for b, p in zip(env.blocks, d["block_pos"][i]):
            b["pos"] = tuple(int(v) for v in p)
        env.t = int(d["t"][i])
        for k in range(K):
            a = env.expert_action()
            if k == 0:
                assert tuple(a) == tuple(int(v) for v in d["action"][i]), "state reconstruction mismatch"
            chunk[j, k] = (a[0] + 1, a[1] + 1, a[2]); mask[j, k] = True
            _, _, done, _ = env.step(a)
            if done:
                break
    return chunk, mask


def build_all(path, K, H, cache=True):
    """Chunk/hist arrays for EVERY row of the file (cached next to it, because rollouts take a while)."""
    cpath = f"{path}.K{K}H{H}.chunks.npz"
    if cache and os.path.exists(cpath) and os.path.getmtime(cpath) >= os.path.getmtime(path):
        c = np.load(cpath)
        return {k: c[k] for k in c.files}
    d = {k: v for k, v in np.load(path, allow_pickle=False).items()}
    N = len(d["t"])
    src = d["ep_source"] if "ep_source" in d else np.array(["expert"] * len(d["ep_length"]))
    chunk, mask = logged_chunks(d, K)                      # default = logged; overwrite expert/dagger rows
    row_src = src[d["episode_id"]]
    rows = np.nonzero(row_src != "human")[0]
    if len(rows):
        c, m = rollout_chunks(d, rows, K)
        chunk[rows], mask[rows] = c, m
    out = dict(chunk=chunk, cmask=mask, hist=effective_history(d, H), gs=d["gripper_closed"].astype(np.int64))
    if cache:
        np.savez_compressed(cpath, **out)
    return out


# ----------------------------------------------------------------------------- loader
def load_chunked(path, K=4, H=4, split=None, episodes=None, cache=True):
    """Same arrays as vla_data.load_expert/load_human plus chunk, cmask, hist, gs.
       split in {'train','val','test'} selects by split; `episodes` (list of episode ids) selects explicitly."""
    from make_dataset import SPLITS
    d = {k: v for k, v in np.load(path, allow_pickle=False).items()}
    if episodes is None:
        episodes = np.nonzero(d["ep_split"] == SPLITS[split])[0] if split else np.arange(len(d["ep_length"]))
    rows = D._rows_of(d, np.asarray(episodes))
    arr = D._arrays(d, rows)
    extra = build_all(path, K, H, cache)
    arr.update({k: v[rows] for k, v in extra.items()})
    return arr


def load_human_chunked(path, K=4, H=4, n_holdout=5):
    d = np.load(path, allow_pickle=False)
    n_ep = len(d["ep_length"]); n_holdout = min(n_holdout, max(n_ep - 1, 0))
    eps = np.arange(n_ep)
    tr = load_chunked(path, K, H, episodes=eps[:n_ep - n_holdout])
    ho = load_chunked(path, K, H, episodes=eps[n_ep - n_holdout:]) if n_holdout else None
    return tr, ho


def concat(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a}


# ----------------------------------------------------------------------------- temporal ensembling (numpy)
def ensemble_probs(entries, t, k_decay):
    """
    entries: list of (t_pred, probs) OLDEST FIRST, probs = tuple(dx (K,3), dy (K,3), grip (K,2)) of one env.
    Returns the weighted-average probabilities for step t using every chunk that covers t
    (ACT-style: weight exp(-k_decay * rank), rank 0 = oldest chunk).  k_decay=0 -> plain mean.
    """
    cover = [(tp, p) for tp, p in entries if 0 <= t - tp < len(p[0])]
    assert cover, "no chunk covers this step"
    w = np.exp(-k_decay * np.arange(len(cover))); w /= w.sum()
    return tuple(sum(wi * p[h][t - tp] for wi, (tp, p) in zip(w, cover)) for h in range(3))


# ----------------------------------------------------------------------------- dense grounding labels
N_TOK = E.G * (E.G + E.MARGIN)               # 12 x 14 = 168 image tokens, one per grid cell (px = 6)


def cell_token(x, y):
    """grid cell (x, y) -> index of the image token that covers it.  x, y: ints, numpy arrays or torch tensors.
    Image row = G-1-y (row 0 is the top / largest y), image column = x + MARGIN, tokens are row-major."""
    return (E.G - 1 - y) * (E.G + E.MARGIN) + (x + E.MARGIN)


def token_cell(tok):
    """inverse of cell_token -> (x, y)"""
    W = E.G + E.MARGIN
    return tok % W - E.MARGIN, E.G - 1 - tok // W