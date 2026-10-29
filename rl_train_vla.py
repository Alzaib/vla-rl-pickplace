"""
rl_train_vla.py -- fine-tune the behavior-cloned VLA with PPO using a SPARSE reward.

    reward = 1 on the step where the task is completed, 0 everywhere else.

    python rl_train_vla.py --init vla_best.pt                  # sparse reward (what you asked for)
    python rl_train_vla.py --init vla_best.pt --n-envs 64      # more episodes per iteration
    python rl_train_vla.py --init vla_best.pt --reward shaped  # DEBUG ONLY: is the code OK when signal is dense?

The recipe (the same skeleton used for RL fine-tuning of big VLAs, at toy scale):
  ACTOR   the BC-pretrained VLA.  Its 3 heads give a categorical distribution over dx, dy, grip, so
          log pi(a|obs) = log p(dx) + log p(dy) + log p(grip).
  CRITIC  a SEPARATE small transformer (same input) that predicts V(obs) = expected discounted success.
  LOOP    1) run the actor in 32+ simulators for `horizon` steps (sampling actions)
          2) GAE advantages: "did this step go better than the critic expected?"
          3) a few PPO epochs: raise the probability of actions with positive advantage, clipped
  Optional --kl-coef keeps the actor close to its BC starting point.

IMPORTANT: with a sparse reward there is learning signal ONLY when an episode succeeds.  If the BC model
succeeds ~3% of the time, a 32-episode iteration contains ~1 success.  Start RL from a BC model that
succeeds at least ~20-30% of the time (more BC data first!), or expect little progress.
Iterations with ZERO successes skip the policy update (only the critic trains) because there is nothing to learn from.
"""
import argparse
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pickplace_env as E
from simple_vla_v2_modified import SimpleVLAv2
from train_vla import build_model, final_eval, get_device, load_ckpt, model_forward
from vec_env import VecEnv, run_episodes
from vla_data import MAX_TEXT_LEN, VOCAB

HEADS = ("dx", "dy", "grip")


# ============================================================================ critic
class VLACritic(SimpleVLAv2):
    """Same body as the VLA, but ends in ONE number: V(obs). Separate weights from the actor."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.value_head = nn.Linear(self.dx_head.in_features, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def value(self, image, tokens, t):
        return self.value_head(self.encode(image, tokens, t)).squeeze(-1)


# ============================================================================ helpers
def upload(images, tokens, t, device):
    """numpy batch -> device tensors.  Images stay uint8 until prep_image()."""
    return (torch.from_numpy(images).to(device), torch.from_numpy(tokens).to(device),
            torch.from_numpy(t).to(device) / E.MAX_STEPS)


def prep_image(u8):
    return u8.permute(0, 3, 1, 2).float() / 255.0


def sample(out, greedy=False):
    """logits dict -> (class ids (B,3), log-prob of the whole 3-token action (B,))"""
    cls, logp = [], 0.0
    for name in HEADS:
        lp = F.log_softmax(out[name], -1)
        a = lp.argmax(-1) if greedy else torch.multinomial(lp.exp(), 1).squeeze(1)
        logp = logp + lp.gather(1, a[:, None]).squeeze(1)
        cls.append(a)
    return torch.stack(cls, 1), logp


def log_probs(out, actions):
    """-> logp of the given actions (B,), entropy (B,), list of full log-softmax tensors (per head)"""
    logp, ent, lps = 0.0, 0.0, []
    for h, name in enumerate(HEADS):
        lp = F.log_softmax(out[name], -1)
        logp = logp + lp.gather(1, actions[:, h:h + 1]).squeeze(1)
        ent = ent - (lp.exp() * lp).sum(-1)
        lps.append(lp)
    return logp, ent, lps


def to_env_actions(cls):
    """class ids (B,3) numpy -> env actions (dx, dy, grip)"""
    return np.stack([cls[:, 0] - 1, cls[:, 1] - 1, cls[:, 2]], 1)


@torch.no_grad()
def evaluate_actor(actor, device, use_time, n=100, greedy=False, seed0=2_000_000):
    was = actor.training
    actor.eval()

    def policy(images, tokens, t):
        img, tok, tt = upload(images, tokens, t, device)
        out = model_forward(actor, prep_image(img), tok, tt, use_time)
        return to_env_actions(sample(out, greedy)[0].cpu().numpy())

    res = run_episodes(policy, n, seed0)
    actor.train(was)
    return res


# ============================================================================ rollouts
@torch.no_grad()
def collect(actor, critic, venv, horizon, device, use_time):
    actor.eval(); critic.eval()
    keep = {k: [] for k in ["img", "tok", "t", "act", "logp", "val", "rew", "done"]}
    n_eps = n_succ = 0
    len_sum = 0
    for _ in range(horizon):
        img_u8, tok, tt = upload(*venv.observe(), device)
        img = prep_image(img_u8)
        out = model_forward(actor, img, tok, tt, use_time)
        act, logp = sample(out)
        val = critic.value(img, tok, tt)
        rew, done, succ, lens = venv.step(to_env_actions(act.cpu().numpy()))
        for k, v in zip(keep, [img_u8, tok, tt, act, logp, val,
                               torch.from_numpy(rew).to(device), torch.from_numpy(done.astype(np.float32)).to(device)]):
            keep[k].append(v)
        n_eps += int(done.sum()); n_succ += int(succ.sum()); len_sum += int(lens.sum())
    img_u8, tok, tt = upload(*venv.observe(), device)
    last_val = critic.value(prep_image(img_u8), tok, tt)
    actor.train(); critic.train()
    data = dict(img=torch.cat(keep["img"]), tok=torch.cat(keep["tok"]), t=torch.cat(keep["t"]),
                act=torch.cat(keep["act"]), logp=torch.stack(keep["logp"]), val=torch.stack(keep["val"]),
                rew=torch.stack(keep["rew"]), done=torch.stack(keep["done"]), last_val=last_val)
    return data, dict(episodes=n_eps, successes=n_succ, mean_len=len_sum / max(n_eps, 1))


def gae(rew, val, done, last_val, gamma, lam):
    """all (T, n) tensors.  Episode ends (success OR timeout) are treated as terminal; the policy sees the
    time step as an input, so the timeout is predictable."""
    T = rew.shape[0]
    adv, g = torch.zeros_like(rew), torch.zeros_like(last_val)
    for t in reversed(range(T)):
        nxt = last_val if t == T - 1 else val[t + 1]
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nxt * nonterm - val[t]
        g = delta + gamma * lam * nonterm * g
        adv[t] = g
    return adv, adv + val


# ============================================================================ PPO update
def ppo_update(actor, critic, aopt, copt, data, ref, args, use_time, update_policy):
    adv, ret = gae(data["rew"], data["val"], data["done"], data["last_val"], args.gamma, args.gae_lambda)
    N = adv.numel()
    adv, ret = adv.reshape(-1), ret.reshape(-1)
    old_logp = data["logp"].reshape(-1)
    if update_policy:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ref_lp = None
    if ref is not None and update_policy and args.kl_coef > 0:
        with torch.no_grad():
            chunks = []
            for i in range(0, N, 256):
                o = model_forward(ref, prep_image(data["img"][i:i + 256]), data["tok"][i:i + 256],
                                  data["t"][i:i + 256], use_time)
                chunks.append([F.log_softmax(o[nm], -1) for nm in HEADS])
            ref_lp = [torch.cat([c[h] for c in chunks]) for h in range(3)]

    stats = dict(vloss=0.0, kl=0.0, clip=0.0, ent=0.0, nb=0, pb=0)
    policy_on = update_policy
    for _ in range(args.ppo_epochs):
        perm = torch.randperm(N, device=adv.device)
        for i in range(0, N, args.minibatch):
            idx = perm[i:i + args.minibatch]
            img, tok, tt = prep_image(data["img"][idx]), data["tok"][idx], data["t"][idx]

            v = critic.value(img, tok, tt)                       # ---- critic
            vloss = F.mse_loss(v, ret[idx])
            copt.zero_grad(set_to_none=True)
            vloss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            copt.step()
            stats["vloss"] += vloss.item(); stats["nb"] += 1
            if not policy_on:
                continue

            out = model_forward(actor, img, tok, tt, use_time)   # ---- actor
            logp, ent, lps = log_probs(out, data["act"][idx])
            ratio = torch.exp(logp - old_logp[idx])
            a = adv[idx]
            pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
            loss = pg - args.ent_coef * ent.mean()
            if ref_lp is not None:
                kl_ref = sum((lp.exp() * (lp - r[idx])).sum(-1) for lp, r in zip(lps, ref_lp)).mean()
                loss = loss + args.kl_coef * kl_ref
            aopt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            aopt.step()
            with torch.no_grad():
                approx_kl = (old_logp[idx] - logp).mean().item()
                stats["kl"] += approx_kl
                stats["clip"] += ((ratio - 1).abs() > args.clip).float().mean().item()
                stats["ent"] += ent.mean().item(); stats["pb"] += 1
            if args.target_kl and approx_kl > args.target_kl:    # policy moved enough this iteration
                policy_on = False
    nb, pb = max(stats["nb"], 1), max(stats["pb"], 1)
    return dict(vloss=stats["vloss"] / nb, kl=stats["kl"] / pb, clip=stats["clip"] / pb, ent=stats["ent"] / pb)


# ============================================================================ main loop
def save_actor(path, actor, cfg, it, extra):
    torch.save(dict(model=actor.state_dict(), config=cfg, epoch=it, extra=extra), path)


def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = get_device(args.device)
    actor, cfg = load_ckpt(args.init, device)
    use_time = cfg["use_time"]
    critic = VLACritic(len(VOCAB), cfg["d_model"], MAX_TEXT_LEN, cfg["n_heads"], args.critic_layers).to(device)
    ref = None
    if args.kl_coef > 0:
        ref = copy.deepcopy(actor).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    aopt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    copt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    venv = VecEnv(args.n_envs, args.reward, seed0=5_000_000 + args.seed * 1_000_000)
    print(f"actor from {args.init} | critic {sum(p.numel() for p in critic.parameters()) / 1e6:.2f}M params | "
          f"{args.n_envs} envs x {args.horizon} steps = {args.n_envs * args.horizon} samples/iter | reward: {args.reward}")

    r = evaluate_actor(actor, device, use_time, args.eval_episodes, greedy=False)
    g = evaluate_actor(actor, device, use_time, args.eval_episodes, greedy=True)
    best = r["success_rate"]
    print(f"iter   0 | START  eval success: sampled {r['success_rate']:.2f}  argmax {g['success_rate']:.2f}")
    save_actor(args.best_path, actor, cfg, 0, dict(success=best))
    total_steps = 0

    for it in range(1, args.iters + 1):
        t0 = time.time()
        data, ep = collect(actor, critic, venv, args.horizon, device, use_time)
        total_steps += args.n_envs * args.horizon
        has_signal = float(data["rew"].abs().sum().item()) > 0
        update_policy = it > args.critic_warmup and has_signal
        st = ppo_update(actor, critic, aopt, copt, data, ref, args, use_time, update_policy)
        note = "" if update_policy else ("  [critic warm-up]" if it <= args.critic_warmup else "  [no reward signal -> policy not updated]")
        print(f"iter {it:3d} | steps {total_steps:7d} | train eps {ep['episodes']:3d} succ {ep['successes']:2d} "
              f"({ep['successes'] / max(ep['episodes'], 1):.2f}) len {ep['mean_len']:.0f} | ent {st['ent']:.2f} "
              f"kl {st['kl']:.4f} clip {st['clip']:.2f} vloss {st['vloss']:.4f} | {time.time() - t0:.0f}s{note}")

        if it % args.eval_every == 0 or it == args.iters:
            r = evaluate_actor(actor, device, use_time, args.eval_episodes, greedy=False)
            g = evaluate_actor(actor, device, use_time, args.eval_episodes, greedy=True)
            print(f"         >> eval ({args.eval_episodes} scenes): sampled {r['success_rate']:.2f}  argmax {g['success_rate']:.2f}")
            if r["success_rate"] > best:
                best = r["success_rate"]
                save_actor(args.best_path, actor, cfg, it, dict(success=best))
                print(f"         >> new best -> {args.best_path}")
    save_actor(args.last_path, actor, cfg, args.iters, dict(success=best))
    final_eval(args.best_path, device, 100)          # 100 NEW scenes, never used for selection


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="vla_best.pt", help="behavior-cloned checkpoint to start from")
    ap.add_argument("--reward", choices=["sparse", "shaped"], default="sparse")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=60, help="steps per env per iteration (60 = one full episode)")
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5, help="actor learning rate (small: we are fine-tuning)")
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--critic-layers", type=int, default=2)
    ap.add_argument("--critic-warmup", type=int, default=3, help="first iterations: train only the critic")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--kl-coef", type=float, default=0.0, help="penalty for drifting from the BC policy (0 = off)")
    ap.add_argument("--target-kl", type=float, default=0.03, help="stop policy updates early if KL exceeds this (0 = off)")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--best-path", default="vla_rl_best.pt")
    ap.add_argument("--last-path", default="vla_rl_last.pt")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
