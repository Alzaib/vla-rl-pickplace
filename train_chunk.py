"""
train_chunk.py -- behavior cloning of an ACTION-CHUNKING VLA:   (obs_t, last H actions) -> [a_t .. a_t+K-1]

    python train_chunk.py --data vla_data.npz --human human_data.npz                 # K=4, H=4 (the new model)
    python train_chunk.py --K 1 --H 0                                                # baseline == SimpleVLAv2
    python train_chunk.py --K 4 --H 0                                                # chunking only, no history
    python train_chunk.py --extra dagger_r1.npz                                      # + DAgger data (chunks are rebuilt automatically)
    python train_chunk.py --eval-only vla_chunk_best.pt                              # test every inference mode

INFERENCE MODES (all evaluated in the simulator on unseen scenes)
    replan1      predict a chunk every step, execute only its FIRST action      (receding horizon, m=1)
    open-loop-m  predict a chunk, execute its first m actions blind, then predict again (m = K, or --execute)
    ensemble     predict every step, average the probabilities of all chunks that cover the current step
                 (ACT-style temporal ensembling, weight exp(-ens_k * rank), rank 0 = oldest)
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import chunk_data as C
import pickplace_env as E
from simple_vla_chunk import SimpleVLAChunk
from train_vla import get_device, lr_at
from vla_data import MAX_TEXT_LEN, VOCAB, tokenize


# ============================================================================ data
class ChunkData:
    def __init__(self, arr, device, store):
        self.device = device
        t = lambda k, dt=None: torch.from_numpy(arr[k]).to(store)
        self.image, self.tokens, self.t = t("image"), t("tokens"), t("t")
        self.chunk, self.cmask, self.hist, self.gs, self.aux = t("chunk"), t("cmask"), t("hist"), t("gs"), t("aux")
        self.n = len(self.t)

    def batch(self, idx):
        idx = idx.to(self.image.device)
        g = lambda x: x[idx].to(self.device)
        img = g(self.image).permute(0, 3, 1, 2).float() / 255.0
        return img, g(self.tokens), g(self.t) / E.MAX_STEPS, g(self.chunk), g(self.cmask), g(self.hist), g(self.aux), g(self.gs)


# ============================================================================ loss / metrics
def grid_loss(out, aux):
    """Per-image-token grounding. Each channel is a softmax over the 168 image tokens:
         ee: the cell under the end-effector | block: the cell of the TARGET block | region: the 3x3 target region
       (region target = uniform over its 9 cells).  This is what teaches 'which patch does the instruction mean'."""
    ee = C.cell_token(aux[:, 0], aux[:, 1])
    blk = C.cell_token(aux[:, 2], aux[:, 3])
    soft = torch.zeros(aux.shape[0], C.N_TOK, device=aux.device)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            soft.scatter_add_(1, C.cell_token(aux[:, 4] + dx, aux[:, 5] + dy)[:, None],
                              torch.full((aux.shape[0], 1), 1.0 / 9, device=aux.device))
    g = out["grid"]
    return (F.cross_entropy(g[:, 0], ee) + F.cross_entropy(g[:, 1], blk)
            - (soft * F.log_softmax(g[:, 2], -1)).sum(-1).mean())


def chunk_loss(out, chunk, cmask, step_w, aux, aux_weight, grid_weight=0.0):
    """Masked cross-entropy over (batch, chunk step, head). Steps past the episode end are ignored."""
    def ce(logits, y):
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none").view_as(y)
    per_step = ce(out["dx"], chunk[..., 0]) + ce(out["dy"], chunk[..., 1]) + ce(out["grip"], chunk[..., 2])   # (B,K)
    w = cmask.float() * step_w[None, :]
    loss = (per_step * w).sum() / w.sum().clamp(min=1.0)
    if aux_weight > 0:
        loss = loss + aux_weight * F.cross_entropy(out["aux"].reshape(-1, out["aux"].shape[-1]), aux.reshape(-1))
    if grid_weight > 0:
        loss = loss + grid_weight * grid_loss(out, aux)
    return loss


@torch.no_grad()
def open_loop(model, data, step_w, aux_weight, grid_weight=0.0, bs=256):
    was = model.training
    model.eval()
    K = model.K
    loss = 0.0
    corr, nval = np.zeros((K, 3)), np.zeros(K)
    all0 = all0_nohist = ev_ok = ev_n = 0
    aux_ok, cell_ok = np.zeros(6), np.zeros(3)
    for i in range(0, data.n, bs):
        idx = torch.arange(i, min(i + bs, data.n))
        img, tok, t, chunk, cmask, hist, aux, gs = data.batch(idx)
        out = model(img, tok, t, hist)
        loss += chunk_loss(out, chunk, cmask, step_w, aux, aux_weight, grid_weight).item() * len(idx)
        pred = torch.stack([out["dx"].argmax(-1), out["dy"].argmax(-1), out["grip"].argmax(-1)], -1)
        ok = pred == chunk
        corr += (ok & cmask[..., None]).sum(0).cpu().numpy()
        nval += cmask.sum(0).cpu().numpy()
        all0 += ok[:, 0].all(-1).sum().item()
        aux_ok += (out["aux"].argmax(-1) == aux).sum(0).cpu().numpy()
        ga = out["grid"].argmax(-1)                                  # (B, 3) best token for [ee, block, region]
        rx, ry = C.token_cell(ga[:, 2])
        cell_ok += np.array([(ga[:, 0] == C.cell_token(aux[:, 0], aux[:, 1])).sum().item(),
                             (ga[:, 1] == C.cell_token(aux[:, 2], aux[:, 3])).sum().item(),
                             ((rx - aux[:, 4]).abs().le(1) & (ry - aux[:, 5]).abs().le(1)).sum().item()])
        ev = chunk[:, 0, 2] != gs                                    # steps where the label flips the gripper
        ev_ok += (ok[:, 0, 2] & ev).sum().item(); ev_n += ev.sum().item()
        if model.H > 0:                                              # how much does it lean on the history?
            o2 = model(img, tok, t, None)
            p2 = torch.stack([o2["dx"][:, 0].argmax(-1), o2["dy"][:, 0].argmax(-1), o2["grip"][:, 0].argmax(-1)], -1)
            all0_nohist += (p2 == chunk[:, 0]).all(-1).sum().item()
    model.train(was)
    acc = corr / np.maximum(nval, 1)[:, None]
    return dict(loss=loss / data.n, acc=acc, all0=all0 / data.n, all0_nohist=all0_nohist / data.n,
                toggle=ev_ok / max(ev_n, 1), aux=aux_ok / data.n, cell=cell_ok / data.n)


# ============================================================================ closed loop
@torch.no_grad()
def closed_loop_chunk(model, device, n=100, execute=1, ens_k=None, greedy=True, seed0=2_000_000, seed=0, trace=False):
    """
    Run n episodes in parallel. `execute`: how many actions of a chunk are performed before re-planning.
    `ens_k` not None: temporal ensembling (plans every step, averages overlapping chunks).
    The action history fed to the model is computed exactly as in training: the displacement that really
    happened (a bump = 0,0) plus the gripper command.
    """
    K, H = model.K, model.H
    assert 1 <= execute <= K
    rng = np.random.default_rng(seed)
    was = model.training
    model.eval()
    envs = [E.PickPlaceEnv("sparse") for _ in range(n)]
    obs = [e.reset(seed=seed0 + i) for i, e in enumerate(envs)]
    toks = np.array([tokenize(o["instruction"]) for o in obs], np.int64)
    hist = np.tile(C.PAD, (n, H, 1))
    entries = [[] for _ in range(n)]
    succ, steps, active = np.zeros(n, bool), np.zeros(n), list(range(n))
    if trace:
        tb0 = [e.blocks[e.target_block]["pos"] for e in envs]
        st = dict(reached=np.zeros(n, bool), grasped=np.zeros(n, bool), carried=np.zeros(n, bool), wrong=np.zeros(n),
                  bumps=np.zeros(n), toggles=np.zeros(n), cells=[set() for _ in range(n)])
        prev_g = [0] * n
    while active:
        now = {i: obs[i]["t"] for i in active}
        need = [i for i in active if ens_k is not None or not entries[i] or now[i] - entries[i][-1][0] >= execute]
        if need:
            img = torch.from_numpy(np.stack([obs[i]["image"] for i in need])).to(device).permute(0, 3, 1, 2).float() / 255.0
            tt = torch.tensor([now[i] / E.MAX_STEPS for i in need], dtype=torch.float32, device=device)
            hh = torch.from_numpy(hist[need]).to(device) if H > 0 else None
            out = model(img, torch.from_numpy(toks[need]).to(device), tt, hh)
            pr = [F.softmax(out[k], -1).cpu().numpy() for k in ("dx", "dy", "grip")]      # (m, K, classes)
            for j, i in enumerate(need):
                entries[i] = (entries[i] + [(now[i], tuple(p[j] for p in pr))])[-K:]
        nxt = []
        for i in active:
            if ens_k is not None:
                pe = C.ensemble_probs(entries[i], now[i], ens_k)
            else:
                tp, p = entries[i][-1]
                pe = tuple(q[now[i] - tp] for q in p)
            cls = [int(q.argmax()) if greedy else int(rng.choice(len(q), p=q / q.sum())) for q in pe]
            ee0 = envs[i].ee
            o, r, d, info = envs[i].step((cls[0] - 1, cls[1] - 1, cls[2]))
            if H > 0:
                eff = [int(np.clip(envs[i].ee[0] - ee0[0], -1, 1)) + 1, int(np.clip(envs[i].ee[1] - ee0[1], -1, 1)) + 1, cls[2]]
                hist[i] = np.concatenate([hist[i][1:], np.array([eff])])
            obs[i] = o
            if trace:
                e = envs[i]
                holding_target = e.held is not None and e.held == e.target_block
                st["cells"][i].add(e.ee); st["reached"][i] |= e.ee == tb0[i]; st["grasped"][i] |= holding_target
                st["carried"][i] |= holding_target and bool(e.region_masks[e.target_region][e.ee])
                st["wrong"][i] += info["wrong_grasp"]; st["bumps"][i] += info["collided"]
                st["toggles"][i] += cls[2] != prev_g[i]; prev_g[i] = cls[2]
            if d:
                succ[i], steps[i] = info["success"], envs[i].t
            else:
                nxt.append(i)
        active = nxt
    model.train(was)
    res = dict(success_rate=float(succ.mean()), mean_steps=float(steps.mean()))
    if trace:
        res["stages"] = dict(reached_block=st["reached"].mean(), grasped_target=st["grasped"].mean(),
                             carried_to_region=st["carried"].mean(), success=succ.mean(),
                             wrong_grasps_per_ep=st["wrong"].mean(), bumps_per_ep=st["bumps"].mean(),
                             gripper_toggles_per_ep=st["toggles"].mean(),
                             unique_cells_per_step=float(np.mean([len(c) / max(s_, 1) for c, s_ in zip(st["cells"], steps)])))
    return res


def eval_modes(model, device, n, full=False, seed0=2_000_000, ens_k=0.1, execute=None):
    K = model.K
    modes = {"replan1": dict(execute=1)}
    if K > 1:
        modes[f"open-loop-{execute or K}"] = dict(execute=execute or K)
        modes["ensemble"] = dict(execute=1, ens_k=ens_k)
    res = {}
    for name, kw in modes.items():
        res[name] = closed_loop_chunk(model, device, n, greedy=True, seed0=seed0, **kw)["success_rate"]
        if full:
            res[name + " (sampled)"] = closed_loop_chunk(model, device, n, greedy=False, seed0=seed0, **kw)["success_rate"]
    return res


# ============================================================================ checkpoints
def save_ckpt(path, model, cfg, epoch, extra=None):
    torch.save(dict(model=model.state_dict(), config=cfg, epoch=epoch, extra=extra or {}), path)


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    c = ck["config"]
    model = SimpleVLAChunk(c["vocab_size"], c["d_model"], c["max_text_len"], c["n_heads"], c["n_layers"],
                           c["K"], c["H"], dropout=c["dropout"], hist_dropout=c["hist_dropout"]).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not unexpected and all(k.startswith("grid_head") for k in missing), (missing, unexpected)
    if missing:
        print("note: checkpoint predates the grid head (its grounding head is untrained)")
    return model, c


# ============================================================================ training
def train(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = get_device(a.device)
    store = torch.device("cpu") if a.data_on_cpu else device
    K, H = a.K, a.H

    print(f"building chunk targets (K={K}, H={H}); first run takes a while, then it is cached next to each file ...")
    train_arr = C.load_chunked(a.data, K, H, "train")
    for p in a.extra:
        ex = C.load_chunked(p, K, H, "train"); train_arr = C.concat(train_arr, ex); print(f"extra {p}: {len(ex['t'])} rows")
    n_pre = len(train_arr["t"])
    val = C.load_chunked(a.data, K, H, "val")
    hold = None
    if a.human:
        h_tr, hold = C.load_human_chunked(a.human, K, H, a.human_holdout)
        train_arr = C.concat(train_arr, h_tr)
        print(f"human: {len(h_tr['t'])} train rows (x{a.human_repeat}), {0 if hold is None else len(hold['t'])} held-out rows")
    print(f"train rows {len(train_arr['t'])} | val rows {len(val['t'])}")
    tr_d, val_d = ChunkData(train_arr, device, store), ChunkData(val, device, store)
    hold_d = ChunkData(hold, device, store) if hold is not None else None
    pool = torch.from_numpy(np.concatenate([np.arange(tr_d.n)] + [np.arange(n_pre, tr_d.n)] * max(a.human_repeat - 1, 0)))

    cfg = dict(vocab_size=len(VOCAB), d_model=a.d_model, max_text_len=MAX_TEXT_LEN, n_heads=a.n_heads,
               n_layers=a.n_layers, K=K, H=H, dropout=a.dropout, hist_dropout=a.hist_dropout)
    model = SimpleVLAChunk(**{k: v for k, v in cfg.items()}).to(device)
    print(f"SimpleVLAChunk K={K} H={H}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    step_w = torch.tensor([a.step_decay ** k for k in range(K)], dtype=torch.float32, device=device)
    spe = math.ceil(len(pool) / a.batch_size); total, step, best = spe * a.epochs, 0, -1.0
    model.train()
    for epoch in range(1, a.epochs + 1):
        t0, perm, run = time.time(), torch.randperm(len(pool)), 0.0
        for i in range(0, len(perm), a.batch_size):
            idx = pool[perm[i:i + a.batch_size]]
            img, tok, t, chunk, cmask, hist, aux, _ = tr_d.batch(idx)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, a.lr)
            loss = chunk_loss(model(img, tok, t, hist), chunk, cmask, step_w, aux, a.aux_weight, a.grid_weight)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            run += loss.item() * len(idx); step += 1
        v = open_loop(model, val_d, step_w, a.aux_weight, a.grid_weight)
        line = (f"epoch {epoch:3d}/{a.epochs} train {run / len(perm):.3f} | val {v['loss']:.3f} | step0 acc dx {v['acc'][0, 0]:.2f} "
                f"dy {v['acc'][0, 1]:.2f} grip {v['acc'][0, 2]:.2f} all {v['all0']:.2f} | grip-toggle acc {v['toggle']:.2f}")
        line += f" | cell acc ee {v['cell'][0]:.2f} block {v['cell'][1]:.2f} region {v['cell'][2]:.2f}"
        line += (f" | where-is-it acc ee {v['aux'][:2].mean():.2f} block {v['aux'][2:4].mean():.2f} "
                 f"region {v['aux'][4:].mean():.2f}")
        if K > 1:
            line += f" | acc(dx,dy) at chunk step {K - 1}: {v['acc'][-1, 0]:.2f},{v['acc'][-1, 1]:.2f}"
        if H > 0:
            line += f" | all w/o history {v['all0_nohist']:.2f}"
        if hold_d is not None:
            line += f" | human-holdout all {open_loop(model, hold_d, step_w, a.aux_weight, a.grid_weight)['all0']:.2f}"
        print(line + f"  [{time.time() - t0:.0f}s]")
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            res = eval_modes(model, device, a.eval_episodes, ens_k=a.ens_k)
            print("   >> closed-loop success (argmax): " + "  ".join(f"{k} {s:.2f}" for k, s in res.items()))
            score = max(res.values())
            if score > best:
                best = score; save_ckpt(a.best_path, model, cfg, epoch, dict(success=score, modes=res))
                print(f"   >> new best -> {a.best_path}")
    save_ckpt(a.last_path, model, cfg, a.epochs)
    final(a.best_path, device, 100, a.ens_k, a.execute)


def final(path, device, n, ens_k, execute):
    model, c = load_ckpt(path, device)
    print(f"\n=== final test of {path} (K={c['K']}, H={c['H']}): {n} NEW scenes ===")
    for k, s in eval_modes(model, device, n, full=True, seed0=2_100_000, ens_k=ens_k, execute=execute).items():
        print(f"  {k:24s} success {s:.2f}")
    r = closed_loop_chunk(model, device, n, execute=1, greedy=True, seed0=2_100_000, trace=True)
    print("\n  WHERE DO EPISODES FAIL? (replan1, argmax; fraction of episodes that reach each stage)")
    for k, v in r["stages"].items():
        print(f"    {k:26s} {v:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="expert_data.npz")
    ap.add_argument("--extra", nargs="*", default=[], help="extra datasets, e.g. dagger_r1.npz")
    ap.add_argument("--human", default=None)
    ap.add_argument("--human-holdout", type=int, default=5)
    ap.add_argument("--human-repeat", type=int, default=1)
    ap.add_argument("--K", type=int, default=4, help="chunk length (actions predicted per forward pass)")
    ap.add_argument("--H", type=int, default=4, help="number of past actions given as input (0 = none)")
    ap.add_argument("--hist-dropout", type=float, default=0.3)
    ap.add_argument("--step-decay", type=float, default=1.0, help="loss weight of chunk step k is decay**k")
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--grid-weight", type=float, default=1.0, help="weight of the per-image-token grounding loss")
    ap.add_argument("--execute", type=int, default=None, help="open-loop steps before re-planning (default K)")
    ap.add_argument("--ens-k", type=float, default=0.1, help="temporal-ensemble decay (0 = plain average)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--data-on-cpu", action="store_true")
    ap.add_argument("--best-path", default="vla_chunk_best.pt")
    ap.add_argument("--last-path", default="vla_chunk_last.pt")
    ap.add_argument("--eval-only", default=None, metavar="CKPT")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.eval_only:
        final(args.eval_only, get_device(args.device), 100, args.ens_k, args.execute)
    else:
        train(args)