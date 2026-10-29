"""
vec_env.py -- many simulators stepped together so the policy can act on a whole batch at once.
(NumPy only, no torch.)

  VecEnv(n)            n parallel PickPlaceEnv's that AUTO-RESET when an episode ends (for RL rollouts)
  run_episodes(fn, n)  run n fixed-seed episodes to completion with a batched policy (for evaluation)

Reward modes come from PickPlaceEnv:
  "sparse": 1.0 on the step where the task succeeds, else 0.0     <- what we use for RL
  "shaped": progress-based reward (only for debugging / comparison)
"""
import numpy as np

import pickplace_env as E
from vla_data import tokenize


class VecEnv:
    def __init__(self, n, reward_mode="sparse", seed0=5_000_000):
        self.n = n
        self.envs = [E.PickPlaceEnv(reward_mode) for _ in range(n)]
        self.next_seed = seed0
        self.obs = [None] * n
        self.tokens = np.zeros((n, 8), np.int64)
        for i in range(n):
            self._reset(i)

    def _reset(self, i):
        self.obs[i] = self.envs[i].reset(seed=self.next_seed)
        self.next_seed += 1
        self.tokens[i] = tokenize(self.obs[i]["instruction"])

    def observe(self):
        """(images uint8 (n,H,W,3), tokens int64 (n,8), t float32 (n,))"""
        return (np.stack([o["image"] for o in self.obs]), self.tokens.copy(),
                np.array([o["t"] for o in self.obs], np.float32))

    def step(self, actions):
        """actions: (n,3) env actions (dx,dy,grip).  Returns rewards, dones, successes, episode_lengths.
        `successes[i]` is True only on the step where episode i ended in success.  After a done, env i is
        immediately reset, so the next observe() shows the FIRST frame of a fresh episode."""
        rewards = np.zeros(self.n, np.float32)
        dones = np.zeros(self.n, bool)
        succ = np.zeros(self.n, bool)
        lens = np.zeros(self.n, np.int64)
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            o, r, d, info = env.step(tuple(int(v) for v in a))
            rewards[i], dones[i] = r, d
            if d:
                succ[i], lens[i] = info["success"], env.t
                self._reset(i)
            else:
                self.obs[i] = o
        return rewards, dones, succ, lens


def run_episodes(policy_batch_fn, n=100, seed0=2_000_000, reward_mode="sparse"):
    """
    Run n episodes (seeds seed0 .. seed0+n-1) to completion.
        policy_batch_fn(images, tokens, t) -> int array (k, 3) of env actions (dx, dy, grip)
    where k = number of episodes still running.  Returns dict(success_rate, mean_steps).
    """
    envs = [E.PickPlaceEnv(reward_mode) for _ in range(n)]
    obs = [e.reset(seed=seed0 + i) for i, e in enumerate(envs)]
    toks = [tokenize(o["instruction"]) for o in obs]
    succ, steps, active = np.zeros(n, bool), np.zeros(n), list(range(n))
    while active:
        acts = policy_batch_fn(np.stack([obs[i]["image"] for i in active]),
                               np.array([toks[i] for i in active], np.int64),
                               np.array([obs[i]["t"] for i in active], np.float32))
        nxt = []
        for k, i in enumerate(active):
            o, r, d, info = envs[i].step(tuple(int(v) for v in acts[k]))
            obs[i] = o
            if d:
                succ[i], steps[i] = info["success"], envs[i].t
            else:
                nxt.append(i)
        active = nxt
    return dict(success_rate=float(succ.mean()), mean_steps=float(steps.mean()))
