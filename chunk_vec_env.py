"""
chunk_vec_env.py -- parallel simulators for RL with the chunking VLA.  (NumPy only, no torch)

Same as vec_env.VecEnv (auto-reset, sparse 0/1 reward) plus the action HISTORY the model may take as input:
`self.hist` is (n, H, 3) class ids of the last H performed actions (oldest first, PAD before the episode starts),
computed exactly as in chunk_data.effective_history / train_chunk.closed_loop_chunk:
    performed action = (real end-effector displacement, grip command)     (a bump counts as 0,0)
It is reset to PAD whenever an episode ends.

REWARD (reward_mode="sparse"): 1.0 on the step where the episode succeeds, 0.0 on every other step.
Because success ends the episode, this is exactly "1 if the whole episode succeeded, else 0".
"""
import numpy as np

import chunk_data as C
from vec_env import VecEnv


class ChunkVecEnv(VecEnv):
    def __init__(self, n, H=0, reward_mode="sparse", seed0=5_000_000):
        super().__init__(n, reward_mode, seed0)
        self.H = H
        self.hist = np.tile(C.PAD, (n, H, 1))

    def step(self, actions):
        """actions (n,3) env actions (dx,dy,grip) -> rewards (n,), dones (n,), successes (n,), episode lengths (n,)"""
        n = self.n
        rewards, dones = np.zeros(n, np.float32), np.zeros(n, bool)
        succ, lens = np.zeros(n, bool), np.zeros(n, np.int64)
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            ee0 = env.ee
            o, r, d, info = env.step(tuple(int(v) for v in a))
            if self.H > 0:
                eff = [int(np.clip(env.ee[0] - ee0[0], -1, 1)) + 1, int(np.clip(env.ee[1] - ee0[1], -1, 1)) + 1, int(a[2])]
                self.hist[i] = np.concatenate([self.hist[i][1:], np.array([eff])])
            rewards[i], dones[i] = r, d
            if d:
                succ[i], lens[i] = info["success"], env.t
                self._reset(i)
                if self.H > 0:
                    self.hist[i] = C.PAD
            else:
                self.obs[i] = o
        return rewards, dones, succ, lens
