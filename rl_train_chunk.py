"""
rl_train_chunk.py -- fine-tune the action-chunking VLA (train_chunk.py / g0.pt) with PPO and a BINARY episode reward.

    reward = 1 on the step where the episode succeeds, 0 on every other step
           = "1 if the whole episode/trajectory succeeded, else 0"     (success ends the episode)

    python rl_train_chunk.py --init g0.pt                       # the real run
    python rl_train_chunk.py --init g0.pt --iters 2 --eval-episodes 10 --eval-every 1     # smoke test
    python rl_train_chunk.py --init g0.pt --reward shaped       # DEBUG ONLY: dense reward, to check the code

HOW THE CHUNK MODEL IS USED IN RL
  The policy being trained is the "replan every step" policy: at every step the model predicts a chunk, but only
  chunk step 0 (the action that is actually executed) is sampled and enters the PPO objective:
        log pi(a | obs, history) = log p_dx + log p_dy + log p_grip      (all from chunk step 0)
  Chunk steps 1..K-1, the coordinate head and the per-token grounding head are not used by RL; they only change
  indirectly because they share the transformer.  (So after RL, evaluate with replan1; open-loop chunk execution
  is not what was optimised.)
  ACTOR   the BC model.   CRITIC  a second copy of the same architecture (K=1) that outputs V(obs, history).
  By default the critic STARTS from the actor's weights (--critic-init actor): it already knows where the block and the
  region are, so it does not have to rediscover grounding from a 0/1 reward.
  Both nets stay in eval mode during training: no dropout, so old/new log-probs are comparable in the PPO ratio.
"""
import argparse
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pickplace_env as E
from chunk_vec_env import ChunkVecEnv
from simple_vla_chunk import SimpleVLAChunk
from train_chunk import closed_loop_chunk, final, load_ckpt
from train_vla import get_device

HEADS = ("dx", "dy", "grip")


# ============================================================================ critic
class VLACritic(SimpleVLAChunk):
    """Same body as the actor with ONE query token; ends in a single number V(obs, history)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.value_head = nn.Linear(self.dx_head.in_features, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def value(self, image, tokens, t, hist=None):
        return self.value_head(self.encode(image, tokens, t, hist)[:, 0]).squeeze(-1)


def build_critic(actor, cfg, device, from_actor=True):
    critic = VLACritic(cfg["vocab_size"], cfg["d_model"], cfg["max_text_len"], cfg["n_heads"], cfg["n_layers"],
                       K=1, H=cfg["H"], dropout=0.0, hist_dropout=0.0).to(device)
    if from_actor:
        own, src = critic.state_dict(), actor.state_dict()
        for k, v in src.items():
            if k.startswith(("dx_head", "dy_head", "grip_head", "aux_head", "grid_head")):
                continue
            if k == "query":
                own[k].copy_(v[:1])                          # critic has a single query token
            elif k in own and own[k].shape == v.shape:
                own[k].copy_(v)
        critic.load_state_dict(own)
    return critic


# ============================================================================ helpers
def upload(images, tokens, t, device):
    """numpy batch -> device tensors. Images stay uint8 until prep_image()."""
    return (torch.from_numpy(images).to(device), torch.from_numpy(tokens).to(device),
            torch.from_numpy(t).to(device) / E.MAX_STEPS)


def prep_image(u8):
    return u8.permute(0, 3, 1, 2).float() / 255.0


def step0(out):
    """chunk logits (B,K,c) -> logits of the action that is executed (chunk step 0)."""
    return {k: out[k][:, 0] for k in HEADS}


def sample(logits):
    """logits dict -> (class ids (B,3), log-prob of the whole 3-head action (B,))"""
    cls, logp = [], 0.0
    for name in HEADS:
        lp = F.log_softmax(logits[name], -1)
        a = torch.multinomial(lp.exp(), 1).squeeze(1)
        logp = logp + lp.gather(1, a[:, None]).squeeze(1)
        cls.append(a)
    return torch.stack(cls, 1), logp


def log_probs(logits, actions):
    """-> logp of the given actions (B,), entropy (B,), per-head full log-softmax"""
    logp, ent, lps = 0.0, 0.0, []
    for h, name in enumerate(HEADS):
        lp = F.log_softmax(logits[name], -1)
        logp = logp + lp.gather(1, actions[:, h:h + 1]).squeeze(1)
        ent = ent - (lp.exp() * lp).sum(-1)
        lps.append(lp)
    return logp, ent, lps


def to_env_actions(cls):
    return np.stack([cls[:, 0] - 1, cls[:, 1] - 1, cls[:, 2]], 1)


def eval_actor(actor, device, n, seed0=2_000_000):
    """closed-loop success of the replan-every-step policy, sampled and argmax, on fixed unseen scenes"""
    s = closed_loop_chunk(actor, device, n, execute=1, greedy=False, seed0=seed0)["success_rate"]
    g = closed_loop_chunk(actor, device, n, execute=1, greedy=True, seed0=seed0)["success_rate"]
    return s, g


# ============================================================================ rollouts
@torch.no_grad()
def collect(actor, critic, venv, horizon, device):
    keep = {k: [] for k in ["img", "tok", "t", "hist", "act", "logp", "val", "rew", "done"]}
    n_eps = n_succ = len_sum = 0
    for _ in range(horizon):
        img_u8, tok, tt = upload(*venv.observe(), device)
        hist = torch.from_numpy(venv.hist.copy()).to(device)
        img = prep_image(img_u8)
        act, logp = sample(step0(actor(img, tok, tt, hist)))
        val = critic.value(img, tok, tt, hist)
        rew, done, succ, lens = venv.step(to_env_actions(act.cpu().numpy()))
        for k, v in zip(keep, [img_u8, tok, tt, hist, act, logp, val, torch.from_numpy(rew).to(device),
                               torch.from_numpy(done.astype(np.float32)).to(device)]):
            keep[k].append(v)
        n_eps += int(done.sum()); n_succ += int(succ.sum()); len_sum += int(lens.sum())
    img_u8, tok, tt = upload(*venv.observe(), device)
    last_val = critic.value(prep_image(img_u8), tok, tt, torch.from_numpy(venv.hist.copy()).to(device))
    data = dict(img=torch.cat(keep["img"]), tok=torch.cat(keep["tok"]), t=torch.cat(keep["t"]),
                hist=torch.cat(keep["hist"]), act=torch.cat(keep["act"]), logp=torch.stack(keep["logp"]),
                val=torch.stack(keep["val"]), rew=torch.stack(keep["rew"]), done=torch.stack(keep["done"]),
                last_val=last_val)
    return data, dict(episodes=n_eps, successes=n_succ, mean_len=len_sum / max(n_eps, 1))


def gae(rew, val, done, last_val, gamma, lam):
    """all (T, n) tensors. Episode ends (success OR timeout) are terminal; the model sees the time step, so the
    timeout is predictable."""
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
def ppo_update(actor, critic, aopt, copt, data, ref, args, update_policy):
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
            for i in range(0, N, 128):
                sl = slice(i, i + 128)
                o = step0(ref(prep_image(data["img"][sl]), data["tok"][sl], data["t"][sl], data["hist"][sl]))
                chunks.append([F.log_softmax(o[nm], -1) for nm in HEADS])
            ref_lp = [torch.cat([c[h] for c in chunks]) for h in range(3)]

    stats = dict(vloss=0.0, kl=0.0, clip=0.0, ent=0.0, nb=0, pb=0)
    policy_on = update_policy
    for _ in range(args.ppo_epochs):
        perm = torch.randperm(N, device=adv.device)
        for i in range(0, N, args.minibatch):
            idx = perm[i:i + args.minibatch]
            img, tok, tt, hist = prep_image(data["img"][idx]), data["tok"][idx], data["t"][idx], data["hist"][idx]

            v = critic.value(img, tok, tt, hist)                  # ---- critic
            vloss = F.mse_loss(v, ret[idx])
            copt.zero_grad(set_to_none=True)
            vloss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            copt.step()
            stats["vloss"] += vloss.item(); stats["nb"] += 1
            if not policy_on:
                continue

            logp, ent, lps = log_probs(step0(actor(img, tok, tt, hist)), data["act"][idx])     # ---- actor
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
            if args.target_kl and approx_kl > args.target_kl:     # policy moved enough this iteration
                policy_on = False
    nb, pb = max(stats["nb"], 1), max(stats["pb"], 1)
    return dict(vloss=stats["vloss"] / nb, kl=stats["kl"] / pb, clip=stats["clip"] / pb, ent=stats["ent"] / pb)


# ============================================================================ main loop
def save_actor(path, actor, cfg, it, extra):
    torch.save(dict(model=actor.state_dict(), config=cfg, epoch=it, extra=extra), path)   # same format as train_chunk.py


def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = get_device(args.device)
    actor, cfg = load_ckpt(args.init, device)
    actor.eval()                                                  # no dropout anywhere during RL
    critic = build_critic(actor, cfg, device, from_actor=(args.critic_init == "actor")).eval()
    ref = None
    if args.kl_coef > 0:
        ref = copy.deepcopy(actor).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    aopt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    copt = torch.optim.Adam([p for n_, p in critic.named_parameters()
                             if not n_.startswith(("dx_head", "dy_head", "grip_head", "aux_head", "grid_head"))],
                            lr=args.critic_lr)
    venv = ChunkVecEnv(args.n_envs, cfg["H"], args.reward, seed0=5_000_000 + args.seed * 1_000_000)
    print(f"actor from {args.init} (K={cfg['K']}, H={cfg['H']}) | critic init: {args.critic_init} | "
          f"{args.n_envs} envs x {args.horizon} steps = {args.n_envs * args.horizon} samples/iter | reward: {args.reward}")

    s, g = eval_actor(actor, device, args.eval_episodes)
    best = s
    print(f"iter   0 | START  eval success ({args.eval_episodes} scenes): sampled {s:.2f}  argmax {g:.2f}")
    save_actor(args.best_path, actor, cfg, 0, dict(success=best))
    total_steps = 0

    for it in range(1, args.iters + 1):
        t0 = time.time()
        data, ep = collect(actor, critic, venv, args.horizon, device)
        total_steps += args.n_envs * args.horizon
        has_signal = float(data["rew"].abs().sum().item()) > 0
        update_policy = it > args.critic_warmup and has_signal
        st = ppo_update(actor, critic, aopt, copt, data, ref, args, update_policy)
        note = "" if update_policy else ("  [critic warm-up]" if it <= args.critic_warmup else "  [no reward -> policy not updated]")
        print(f"iter {it:3d} | steps {total_steps:7d} | train eps {ep['episodes']:3d} succ {ep['successes']:3d} "
              f"({ep['successes'] / max(ep['episodes'], 1):.2f}) len {ep['mean_len']:.0f} | ent {st['ent']:.2f} "
              f"kl {st['kl']:.4f} clip {st['clip']:.2f} vloss {st['vloss']:.4f} | {time.time() - t0:.0f}s{note}")

        if it % args.eval_every == 0 or it == args.iters:
            s, g = eval_actor(actor, device, args.eval_episodes)
            print(f"         >> eval ({args.eval_episodes} scenes): sampled {s:.2f}  argmax {g:.2f}")
            if s > best:
                best = s
                save_actor(args.best_path, actor, cfg, it, dict(success=best))
                print(f"         >> new best (sampled) -> {args.best_path}")
    save_actor(args.last_path, actor, cfg, args.iters, dict(success=best))
    final(args.best_path, device, 100, 0.1, None)               # 100 NEW scenes, never used for selection


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="g0_chunk.pt", help="behavior-cloned chunk checkpoint to start from")
    ap.add_argument("--reward", choices=["sparse", "shaped"], default="sparse")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=60, help="steps per env per iteration (60 = one full episode)")
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5, help="actor learning rate (small: we are fine-tuning)")
    ap.add_argument("--critic-lr", type=float, default=1e-4)
    ap.add_argument("--critic-init", choices=["actor", "scratch"], default="actor")
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
    ap.add_argument("--best-path", default="vla_rl_chunk_best.pt")
    ap.add_argument("--last-path", default="vla_rl_chunk_last.pt")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
