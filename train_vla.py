"""
train_vla.py -- behavior-cloning training of SimpleVLA on the pick-and-place dataset.

    python train_vla.py                                          # model v2 + auxiliary loss, expert data only
    python train_vla.py --arch v1 --aux-weight 0                 # the original skeleton
    python train_vla.py --human human_data.npz                   # + your human demos
    python train_vla.py --human human_data.npz --human-repeat 5  # show human rows 5x per epoch
    python train_vla.py --eval-only vla_best.pt                  # just test a saved model

What happens
  1. Load arrays (image, instruction tokens, time, action classes) and put them on the GPU.
  2. Each step: batch -> SimpleVLA -> 3 cross-entropy losses (dx, dy, grip) -> AdamW.
  3. Each epoch: open-loop metrics (how often did we predict the expert's action?).
  4. Every --eval-every epochs: CLOSED-LOOP test = run the model in the simulator on unseen scenes.
     This is the number that matters.  Best model so far is saved to vla_best.pt.
  5. At the end: 100-episode closed-loop test of the best model, argmax and sampled actions.
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import pickplace_env as E
from simple_vla import SimpleVLA
from simple_vla_v2 import SimpleVLAv2
from vla_data import (MAX_TEXT_LEN, VOCAB, concat, decode_action, load_expert, load_human, n_rows, tokenize)


# ============================================================================ device
def get_device(choice="auto"):
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False. "
                             "See the GPU setup notes (CPU-only PyTorch build, or driver problem).")
        p = torch.cuda.get_device_properties(0)
        print(f"device: cuda  ({p.name}, {p.total_memory / 2**30:.1f} GB)")
    else:
        print("device: cpu   (training will work but be much slower)")
    return torch.device(choice)


# ============================================================================ data on device
class DeviceData:
    """Whole dataset lives as tensors (images stay uint8 -> ~280 MB for 15k frames)."""

    def __init__(self, arrays, device, store_device=None):
        sd = store_device or device
        self.device = device
        self.image = torch.from_numpy(arrays["image"]).to(sd)                 # uint8 (N,H,W,3)
        self.tokens = torch.from_numpy(arrays["tokens"]).to(sd)
        self.t = torch.from_numpy(arrays["t"]).to(sd)
        self.action = torch.from_numpy(arrays["action"]).to(sd)
        self.aux = torch.from_numpy(arrays["aux"]).to(sd)
        self.n = len(self.t)

    def batch(self, idx):
        idx = idx.to(self.image.device)
        img = self.image[idx].to(self.device).permute(0, 3, 1, 2).float() / 255.0
        return (img, self.tokens[idx].to(self.device), self.t[idx].to(self.device) / E.MAX_STEPS,
                self.action[idx].to(self.device), self.aux[idx].to(self.device))


# ============================================================================ loss / metrics
def model_forward(model, img, tok, t, use_time):
    return model(img, tok, t if use_time else torch.zeros_like(t))


def bc_loss(out, act):
    return (F.cross_entropy(out["dx"], act[:, 0]) + F.cross_entropy(out["dy"], act[:, 1])
            + F.cross_entropy(out["grip"], act[:, 2]))


def total_loss(out, act, aux, aux_weight):
    loss = bc_loss(out, act)
    if aux_weight > 0 and "aux" in out:
        loss = loss + aux_weight * F.cross_entropy(out["aux"].reshape(-1, out["aux"].shape[-1]), aux.reshape(-1))
    return loss


@torch.no_grad()
def open_loop(model, data, use_time, bs=512):
    """Teacher-forced metrics: given the expert's observation, did we predict the expert's action?"""
    was_training = model.training
    model.eval()
    loss, correct, all_ok = 0.0, np.zeros(3), 0
    for i in range(0, data.n, bs):
        idx = torch.arange(i, min(i + bs, data.n))
        img, tok, t, act, _ = data.batch(idx)
        out = model_forward(model, img, tok, t, use_time)
        loss += bc_loss(out, act).item() * len(idx)
        pred = torch.stack([out["dx"].argmax(1), out["dy"].argmax(1), out["grip"].argmax(1)], 1)
        ok = pred == act
        correct += ok.sum(0).cpu().numpy()
        all_ok += ok.all(1).sum().item()
    model.train(was_training)
    return dict(loss=loss / data.n, dx=correct[0] / data.n, dy=correct[1] / data.n,
                grip=correct[2] / data.n, all=all_ok / data.n)


# ============================================================================ closed loop (simulator)
class VLAPolicy:
    """Wraps a model as policy_fn(image, instruction, t) -> (dx, dy, grip) for E.evaluate / RL later."""

    def __init__(self, model, device, greedy=False, use_time=True):
        self.model, self.device, self.greedy, self.use_time = model, device, greedy, use_time

    @torch.no_grad()
    def __call__(self, image, instruction, t):
        img = torch.from_numpy(image).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tok = torch.tensor([tokenize(instruction)], dtype=torch.long, device=self.device)
        tt = torch.tensor([t / E.MAX_STEPS], dtype=torch.float32, device=self.device)
        out = model_forward(self.model, img, tok, tt, self.use_time)
        picks = []
        for name in ("dx", "dy", "grip"):
            logits = out[name][0]
            picks.append(int(logits.argmax()) if self.greedy else int(torch.multinomial(F.softmax(logits, -1), 1)))
        return decode_action(*picks)


def closed_loop(model, device, n, greedy, use_time, seed0):
    was_training = model.training
    model.eval()
    res = E.evaluate(VLAPolicy(model, device, greedy, use_time), n_episodes=n, seed0=seed0)
    model.train(was_training)
    return res


# ============================================================================ checkpoints
def save_ckpt(path, model, args, epoch, extra=None):
    cfg = dict(arch=args.arch, vocab_size=len(VOCAB), d_model=args.d_model, max_text_len=MAX_TEXT_LEN,
               n_heads=args.n_heads, n_layers=args.n_layers, use_time=not args.no_time, dropout=args.dropout)
    torch.save(dict(model=model.state_dict(), config=cfg, epoch=epoch, extra=extra or {}), path)


def build_model(c):
    if c.get("arch", "v1") == "v2":
        return SimpleVLAv2(c["vocab_size"], c["d_model"], c["max_text_len"], c["n_heads"], c["n_layers"],
                           dropout=c.get("dropout", 0.0))
    return SimpleVLA(c["vocab_size"], c["d_model"], c["max_text_len"], c["n_heads"], c["n_layers"])


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    c = ck["config"]
    model = build_model(c).to(device)
    model.load_state_dict(ck["model"])
    return model, c


# ============================================================================ training
def lr_at(step, total, base_lr, warmup=200):
    warm = min(1.0, (step + 1) / warmup)
    cos = 0.5 * (1 + math.cos(math.pi * min(step / max(total, 1), 1.0)))
    return base_lr * warm * (0.05 + 0.95 * cos)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device)
    store = torch.device("cpu") if args.data_on_cpu else device

    # ---------------- data
    expert_tr = load_expert(args.data, "train", args.expert_episodes)
    expert_val = load_expert(args.data, "val")
    train_arrays, n_expert = expert_tr, n_rows(expert_tr)
    human_hold = None
    if args.human:
        human_tr, human_hold = load_human(args.human, args.human_holdout)
        train_arrays = concat(expert_tr, human_tr)
        print(f"human data: {n_rows(human_tr)} train rows (x{args.human_repeat} per epoch), "
              f"{0 if human_hold is None else n_rows(human_hold)} held-out rows")
    print(f"expert train rows: {n_expert}   expert val rows: {n_rows(expert_val)}   total train rows: {n_rows(train_arrays)}")
    train_data = DeviceData(train_arrays, device, store)
    val_data = DeviceData(expert_val, device, store)
    hold_data = DeviceData(human_hold, device, store) if human_hold is not None else None

    idx_pool = np.concatenate([np.arange(train_data.n)] +
                              [np.arange(n_expert, train_data.n)] * max(args.human_repeat - 1, 0))
    idx_pool = torch.from_numpy(idx_pool)

    # ---------------- model / optimiser
    cfg = dict(arch=args.arch, vocab_size=len(VOCAB), d_model=args.d_model, max_text_len=MAX_TEXT_LEN,
               n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout)
    model = build_model(cfg).to(device)
    print(f"model {args.arch}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters"
          f"   aux loss weight {args.aux_weight if args.arch == 'v2' else 0}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(idx_pool) / args.batch_size)
    total_steps, step, best = steps_per_epoch * args.epochs, 0, -1.0
    use_time = not args.no_time
    model.train()

    for epoch in range(1, args.epochs + 1):
        t0, perm, run_loss = time.time(), torch.randperm(len(idx_pool)), 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = idx_pool[perm[i:i + args.batch_size]]
            img, tok, t, act, aux = train_data.batch(idx)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, args.lr)
            loss = total_loss(model_forward(model, img, tok, t, use_time), act, aux, args.aux_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item() * len(idx)
            step += 1
        v = open_loop(model, val_data, use_time)
        line = (f"epoch {epoch:3d}/{args.epochs}  train loss {run_loss / len(perm):.3f} | val loss {v['loss']:.3f} "
                f"acc dx {v['dx']:.2f} dy {v['dy']:.2f} grip {v['grip']:.2f} all {v['all']:.2f}")
        if hold_data is not None:
            h = open_loop(model, hold_data, use_time)
            line += f" | human-holdout all {h['all']:.2f}"
        print(line + f"  [{time.time() - t0:.0f}s]")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            g = closed_loop(model, device, args.eval_episodes, True, use_time, 2_000_000)
            s = closed_loop(model, device, args.eval_episodes, False, use_time, 2_000_000)
            print(f"   >> closed-loop success ({args.eval_episodes} unseen scenes): "
                  f"argmax {g['success_rate']:.2f}   sampled {s['success_rate']:.2f}")
            score = max(g["success_rate"], s["success_rate"])
            if score > best:
                best = score
                save_ckpt(args.best_path, model, args, epoch, dict(success=score))
                print(f"   >> new best -> {args.best_path}")
    save_ckpt(args.last_path, model, args, args.epochs)
    print(f"saved {args.last_path}")
    final_eval(args.best_path, device, 100)


def final_eval(path, device, n):
    model, c = load_ckpt(path, device)
    use_time = c["use_time"]
    print(f"\n=== final test of {path}: {n} NEW scenes (never used for model selection) ===")
    for greedy in (True, False):
        r = closed_loop(model, device, n, greedy, use_time, 2_100_000)
        print(f"  {'argmax ' if greedy else 'sampled'}: success {r['success_rate']:.2f}   "
              f"mean steps {r['mean_steps']:.1f}   mean return {r['mean_return']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="expert_data_big.npz")
    ap.add_argument("--human", default=None, help="human demo file, e.g. human_data.npz")
    ap.add_argument("--human-holdout", type=int, default=5, help="human episodes kept out for testing")
    ap.add_argument("--human-repeat", type=int, default=1, help="how many times per epoch each human row is used")
    ap.add_argument("--expert-episodes", type=int, default=None, help="use only the first N expert train episodes")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--arch", choices=["v1", "v2"], default="v2", help="v1 = your skeleton, v2 = cell-aligned tokens")
    ap.add_argument("--aux-weight", type=float, default=0.5, help="weight of the auxiliary position loss (v2 only)")
    ap.add_argument("--dropout", type=float, default=0.0, help="transformer dropout (v2 only)")
    ap.add_argument("--no-time", action="store_true", help="ablation: feed zeros instead of the time step")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--data-on-cpu", action="store_true", help="keep the dataset in RAM instead of GPU memory")
    ap.add_argument("--best-path", default="vla_best.pt")
    ap.add_argument("--last-path", default="vla_last.pt")
    ap.add_argument("--eval-only", default=None, metavar="CKPT")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.eval_only:
        final_eval(a.eval_only, get_device(a.device), 100)
    else:
        train(a)