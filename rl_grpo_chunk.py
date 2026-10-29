"""
rl_grpo_chunk.py -- RL fine-tuning of the chunking VLA with a STRICTLY BINARY episode reward (v2, replaces rl_train_chunk.py).

    reward of an episode = 1 if the task was completed successfully, 0 otherwise.   Nothing else is ever rewarded or
    penalised (no shaping, no step cost, no collision penalty).

    python rl_grpo_chunk.py --init g0.pt                                   # the real run
    python rl_grpo_chunk.py --init g0.pt --iters 2 --scenes 4 --group 4 --eval-episodes 10 --eval-every 1   # smoke test

WHAT CHANGED vs rl_train_chunk.py (and why -- see the chat answer for the long version)
  1. GROUP-RELATIVE, OUTCOME-LEVEL ADVANTAGE (GRPO / RLOO style, as in SimpleVLA-RL).
       Each iteration picks S scenes and rolls out G independent SAMPLED attempts of EACH scene.
       advantage of an attempt = (its 0/1 result - mean result of the G attempts on the same scene) / std.
       Every step of an attempt gets that one number.   "Did THIS attempt beat what the policy usually does on THIS
       scene?"  Scene difficulty cancels out; no learned critic, no GAE, no gamma -> far less noise than before.
  2. DYNAMIC SAMPLING: scenes where all G attempts succeed (or all fail) have zero advantage and are dropped from the
       update (they carry no learning signal, only dilute the batch).
  3. HARD-SCENE REPLAY: scenes where the policy sometimes fails are remembered and re-drawn, weighted toward scenes
       with mixed outcomes (the ones that teach something).  Half of every batch is still FRESH scenes.
  4. CLIP-HIGHER: asymmetric PPO clip (0.2 below, 0.28 above) so rare successful behaviour can grow faster.
  5. KEEP-THE-REST-OF-THE-MODEL-STABLE: RL only touches chunk step 0, so a KL term to the frozen BC model is applied to
       chunk steps 1..K-1 and to the per-token grounding head, otherwise they silently drift (that is why argmax,
       open-loop-4 and ensemble numbers got WORSE last time).  A small KL on step 0 is optional (--kl0).
  6. BIGGER, LESS NOISY EVALUATION: 200 scenes by default; best checkpoint chosen on sampled success, argmax reported.

The policy is still the "replan every step" policy: only the executed action (chunk step 0) enters the PPO objective.
Evaluate the result with replan1 (execute=1).
"""
import argparse
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

import pickplace_env as E
import chunk_data as C
from simple_vla_chunk import SimpleVLAChunk  # noqa: F401  (needed so load_ckpt can rebuild the model)
from train_chunk import closed_loop_chunk, final, load_ckpt
from train_vla import get_device
from vla_data import tokenize

HEADS = ("dx", "dy", "grip")


# ============================================================================ numpy side (testable without torch)
class ScenePool:
    """Fresh scene seeds + a memory of scenes the policy has trouble with."""

    def __init__(self, seed0, cap=300):
        self.next, self.hard, self.cap = seed0, {}, cap          # hard: seed -> success rate of the last G attempts

    def draw(self, S, hard_frac, rng):
        k = min(int(round(S * hard_frac)), len(self.hard))
        hard = []
        if k > 0:
            seeds = np.array(list(self.hard.keys()))
            rates = np.array([self.hard[s] for s in seeds])
            w = 4 * rates * (1 - rates) + 0.1                      # mixed outcomes teach the most; all-fail still revisited
            hard = list(rng.choice(seeds, size=k, replace=False, p=w / w.sum()))
        n_fresh = S - k
        fresh = list(range(self.next, self.next + n_fresh))
        self.next += n_fresh
        seeds = [int(s) for s in hard] + fresh
        return seeds, [False] * len(hard) + [True] * n_fresh

    def update(self, seeds, rates):
        for s, r in zip(seeds, rates):
            if r < 1.0:
                self.hard[int(s)] = float(r)
            else:
                self.hard.pop(int(s), None)                        # solved reliably -> forget it
        if len(self.hard) > self.cap:                              # keep the scenes with the lowest success rate
            keep = sorted(self.hard.items(), key=lambda kv: kv[1])[:self.cap]
            self.hard = dict(keep)


def group_advantages(R, S, G, std_norm=True):
    """R (S*G,) episode results in {0,1}, ordered scene-major (G attempts of scene 0, then scene 1, ...).
    Returns per-episode advantage (S*G,), informative mask per episode (S*G,), per-scene success rate (S,)."""
    Rg = np.asarray(R, np.float64).reshape(S, G)
    mean, std = Rg.mean(1, keepdims=True), Rg.std(1, keepdims=True)
    informative = (std[:, 0] > 1e-8)
    adv = Rg - mean
    if std_norm:
        adv = adv / (std + 1e-6)
    adv = np.where(informative[:, None], adv, 0.0)
    return adv.reshape(-1), np.repeat(informative, G), mean[:, 0]


def rollout(policy_fn, seeds, G, H, max_steps=E.MAX_STEPS):
    """Run G independent attempts on each scene seed, to completion (no auto-reset).
    policy_fn(images_u8 (k,72,84,3), tokens (k,8), t (k,), hist (k,H,3)) -> (class ids (k,3) in {0..2,0..2,0..1}, logp (k,))
    Returns (steps: dict of numpy arrays, one row per executed step, R: (S*G,) 0/1 episode results, lens)."""
    seeds_rep = np.repeat(np.asarray(seeds), G)
    n = len(seeds_rep)
    envs = [E.PickPlaceEnv("sparse") for _ in range(n)]
    obs = [e.reset(seed=int(s)) for e, s in zip(envs, seeds_rep)]
    toks = np.stack([tokenize(o["instruction"]) for o in obs]).astype(np.int64)
    hist = np.tile(C.PAD, (n, H, 1)).astype(np.int64)
    R, lens = np.zeros(n, np.float32), np.zeros(n, np.int64)
    rec = {k: [] for k in ("img", "tok", "t", "hist", "act", "logp", "ep")}
    active = list(range(n))
    while active:
        img = np.stack([obs[i]["image"] for i in active])
        t = np.array([obs[i]["t"] for i in active], np.float32)
        h = hist[active].copy()
        cls, logp = policy_fn(img, toks[active], t, h)
        for k, v in zip(rec, (img, toks[active], t, h, cls, logp, np.array(active))):
            rec[k].append(v)
        nxt = []
        for k, i in enumerate(active):
            env, c = envs[i], cls[k]
            ee0 = env.ee
            a = (int(c[0]) - 1, int(c[1]) - 1, int(c[2]))
            o, r, d, info = env.step(a)
            if H > 0:
                eff = [int(np.clip(env.ee[0] - ee0[0], -1, 1)) + 1, int(np.clip(env.ee[1] - ee0[1], -1, 1)) + 1, int(c[2])]
                hist[i] = np.concatenate([hist[i][1:], np.array([eff])])
            if d:
                R[i] = 1.0 if info["success"] else 0.0            # the ONLY reward: 1 iff the episode succeeded
                lens[i] = env.t
            else:
                obs[i] = o
                nxt.append(i)
        active = nxt
    steps = {k: np.concatenate(v) for k, v in rec.items()}
    return steps, R, lens


# ============================================================================ torch side
def upload(images, tokens, t, device):
    return (torch.from_numpy(images).to(device), torch.from_numpy(tokens).to(device),
            torch.from_numpy(t).to(device) / E.MAX_STEPS)


def prep_image(u8):
    return u8.permute(0, 3, 1, 2).float() / 255.0


def step0(out):
    return {k: out[k][:, 0] for k in HEADS}


def sample(logits):
    cls, logp = [], 0.0
    for name in HEADS:
        lp = F.log_softmax(logits[name], -1)
        a = torch.multinomial(lp.exp(), 1).squeeze(1)
        logp = logp + lp.gather(1, a[:, None]).squeeze(1)
        cls.append(a)
    return torch.stack(cls, 1), logp


def log_probs(logits, actions):
    logp, ent = 0.0, 0.0
    for h, name in enumerate(HEADS):
        lp = F.log_softmax(logits[name], -1)
        logp = logp + lp.gather(1, actions[:, h:h + 1]).squeeze(1)
        ent = ent - (lp.exp() * lp).sum(-1)
    return logp, ent


def make_policy_fn(actor, device):
    @torch.no_grad()
    def fn(img, tok, t, hist):
        im, tk, tt = upload(img, tok, t, device)
        cls, logp = sample(step0(actor(prep_image(im), tk, tt, torch.from_numpy(hist).to(device))))
        return cls.cpu().numpy(), logp.cpu().numpy()
    return fn


def kl_to_ref(out, ref_out):
    """KL(pi || ref) split into (chunk step 0, chunk steps 1..K-1 + grounding map).  Means over the batch."""
    kl_heads = 0.0
    for name in HEADS:
        lp, lr = F.log_softmax(out[name], -1), F.log_softmax(ref_out[name], -1)
        kl_heads = kl_heads + (lp.exp() * (lp - lr)).sum(-1)               # (B, K)
    kl0 = kl_heads[:, 0].mean()
    rest = kl_heads[:, 1:].mean() if kl_heads.shape[1] > 1 else out["dx"].new_zeros(())
    if "grid" in out and "grid" in ref_out:
        lp, lr = F.log_softmax(out["grid"], -1), F.log_softmax(ref_out["grid"], -1)
        rest = rest + (lp.exp() * (lp - lr)).sum(-1).mean()
    return kl0, rest


def ppo_update(actor, ref, opt, steps, adv, args, device):
    """steps: numpy dict restricted to informative episodes; adv: per-step advantage (numpy)."""
    img_u8 = torch.from_numpy(steps["img"]); tok = torch.from_numpy(steps["tok"]); t = torch.from_numpy(steps["t"])
    hist = torch.from_numpy(steps["hist"]); act = torch.from_numpy(steps["act"]).long()
    old_logp = torch.from_numpy(steps["logp"]).float(); adv = torch.from_numpy(adv).float()
    N = adv.numel()
    st = dict(kl=0.0, clip=0.0, ent=0.0, klref0=0.0, klref_rest=0.0, nb=0)
    stop = False
    for _ in range(args.ppo_epochs):
        perm = torch.randperm(N)
        for i in range(0, N, args.minibatch):
            idx = perm[i:i + args.minibatch]
            im, tk = prep_image(img_u8[idx].to(device)), tok[idx].to(device)
            tt, h = t[idx].to(device) / E.MAX_STEPS, hist[idx].to(device)
            a, olp, ad = act[idx].to(device), old_logp[idx].to(device), adv[idx].to(device)
            out = actor(im, tk, tt, h)
            logp, ent = log_probs(step0(out), a)
            ratio = torch.exp(logp - olp)
            pg = -torch.min(ratio * ad, torch.clamp(ratio, 1 - args.clip_low, 1 + args.clip_high) * ad).mean()
            loss = pg - args.ent_coef * ent.mean()
            klref0 = klref_rest = torch.zeros((), device=device)
            if ref is not None:
                with torch.no_grad():
                    ro = ref(im, tk, tt, h)
                klref0, klref_rest = kl_to_ref(out, ro)
                loss = loss + args.kl0 * klref0 + args.keep_coef * klref_rest
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            opt.step()
            with torch.no_grad():
                akl = (olp - logp).mean().item()
                st["kl"] += akl; st["ent"] += ent.mean().item(); st["nb"] += 1
                st["clip"] += ((ratio - 1).abs() > args.clip_low).float().mean().item()
                st["klref0"] += float(klref0); st["klref_rest"] += float(klref_rest)
            if args.target_kl and akl > args.target_kl:
                stop = True
                break
        if stop:
            break
    nb = max(st.pop("nb"), 1)
    return {k: v / nb for k, v in st.items()}


def eval_actor(actor, device, n, seed0=2_000_000):
    s = closed_loop_chunk(actor, device, n, execute=1, greedy=False, seed0=seed0)["success_rate"]
    g = closed_loop_chunk(actor, device, n, execute=1, greedy=True, seed0=seed0)["success_rate"]
    return s, g


def save_actor(path, actor, cfg, it, extra):
    torch.save(dict(model=actor.state_dict(), config=cfg, epoch=it, extra=extra), path)     # same format as train_chunk.py


def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device(args.device)
    actor, cfg = load_ckpt(args.init, device)
    actor.eval()                                                    # no dropout during RL: old/new log-probs comparable
    ref = None
    if args.kl0 > 0 or args.keep_coef > 0:
        ref = copy.deepcopy(actor).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    pool = ScenePool(5_000_000 + args.seed * 1_000_000)
    policy_fn = make_policy_fn(actor, device)
    H, S, G = cfg["H"], args.scenes, args.group
    print(f"actor from {args.init} (K={cfg['K']}, H={H}) | {S} scenes x {G} attempts = {S * G} episodes/iter | "
          f"reward: 1 iff episode succeeds, else 0 | hard-scene fraction {args.hard_frac} | clip {args.clip_low}/{args.clip_high} "
          f"| kl0 {args.kl0} keep {args.keep_coef}")

    s, g = eval_actor(actor, device, args.eval_episodes)
    best = s
    print(f"iter   0 | START  eval success ({args.eval_episodes} scenes): sampled {s:.3f}  argmax {g:.3f}")
    save_actor(args.best_path, actor, cfg, 0, dict(success=best))
    total_eps = 0

    for it in range(1, args.iters + 1):
        t0 = time.time()
        seeds, fresh = pool.draw(S, args.hard_frac if it > 1 else 0.0, rng)
        steps, R, lens = rollout(policy_fn, seeds, G, H)
        total_eps += S * G
        adv_ep, inf_ep, rate = group_advantages(R, S, G, std_norm=not args.no_std_norm)
        pool.update(seeds, rate)
        fresh_rate = float(np.mean([r for r, f in zip(rate, fresh) if f])) if any(fresh) else float("nan")
        n_inf = int(inf_ep.reshape(S, G)[:, 0].sum())
        msg = (f"iter {it:3d} | eps {total_eps:6d} | train succ {R.mean():.2f} (fresh scenes {fresh_rate:.2f}) | "
               f"mixed scenes {n_inf}/{S} | hard pool {len(pool.hard)} | len {lens.mean():.0f}")
        if n_inf > 0:
            keep = inf_ep[steps["ep"]]                              # steps that belong to informative scenes
            sub = {k: v[keep] for k, v in steps.items()}
            adv_step = adv_ep[sub["ep"]]
            st = ppo_update(actor, ref, opt, sub, adv_step, args, device)
            msg += (f" | steps {len(adv_step)} ent {st['ent']:.2f} kl {st['kl']:.4f} clip {st['clip']:.2f} "
                    f"klref0 {st['klref0']:.4f} klref_rest {st['klref_rest']:.4f}")
        else:
            msg += " | no mixed scenes -> no update"
        print(msg + f" | {time.time() - t0:.0f}s")

        if it % args.eval_every == 0 or it == args.iters:
            s, g = eval_actor(actor, device, args.eval_episodes)
            print(f"         >> eval ({args.eval_episodes} scenes): sampled {s:.3f}  argmax {g:.3f}")
            if s > best:
                best = s
                save_actor(args.best_path, actor, cfg, it, dict(success=best))
                print(f"         >> new best (sampled) -> {args.best_path}")
    save_actor(args.last_path, actor, cfg, args.iters, dict(success=best))
    print("\n=== FINAL TEST of the best checkpoint on NEW scenes ===")
    final(args.best_path, device, args.final_episodes, 0.1, None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="g0.pt", help="behavior-cloned chunk checkpoint to start from")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--scenes", type=int, default=24, help="S: distinct scenes per iteration")
    ap.add_argument("--group", type=int, default=8, help="G: independent attempts per scene")
    ap.add_argument("--hard-frac", type=float, default=0.5, help="fraction of each batch drawn from remembered hard scenes")
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--clip-low", type=float, default=0.2)
    ap.add_argument("--clip-high", type=float, default=0.28, help="clip-higher: >clip-low lets good actions grow faster")
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--kl0", type=float, default=0.0, help="KL to the BC model on the executed step (0 = off)")
    ap.add_argument("--keep-coef", type=float, default=1.0, help="KL to BC model on chunk steps 1..K-1 + grounding head")
    ap.add_argument("--target-kl", type=float, default=0.02, help="stop updating early if KL exceeds this (0 = off)")
    ap.add_argument("--no-std-norm", action="store_true", help="advantage = R - group mean (RLOO) instead of /std (GRPO)")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--final-episodes", type=int, default=200)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--best-path", default="vla_grpo_chunk_best.pt")
    ap.add_argument("--last-path", default="vla_grpo_chunk_last.pt")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
