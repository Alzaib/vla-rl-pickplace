"""
pickplace_env.py -- a tiny 2-D pick-and-place world for learning VLA + RL.   (NumPy only)

WORLD
  A G x G table seen from above. A robot base sits beside the table (left). The end-effector
  (EE) lives in a grid cell (x, y). The table holds 3 coloured blocks, 1 container, 1 goal zone
  (both 3x3 cells, different colours) and a few wall obstacles.

WHAT THE POLICY GETS (observation)  -- same inputs as a VLA, nothing more
  image       uint8 (H, W, 3)   RGB camera picture of the scene (EE + gripper state are drawn in it)
  instruction str               e.g. "put the red block in the blue container"
  t           int               step index in the episode      (time_s = t * DT)

WHAT THE POLICY OUTPUTS (action)   action = (dx, dy, grip)
  dx, dy in {-1, 0, +1}   move the EE by that many cells (blocked by obstacles / table edge)
  grip   in {0, 1}        0 = open gripper, 1 = close gripper
  Closing while on top of a block grasps it; opening releases it.

SUCCESS: the named block is released inside the named region.

STATE (privileged ground truth, for supervision / debugging; NOT given to the policy)
  env.state()  -> EE position, gripper, which block is held, all block positions, ...

IMAGE CONVENTION
  Row 0 is the TOP of the picture = largest y. Column 0 is the far-left (robot base side).
  Cell (x, y) is drawn at row (G-1-y), column (x + MARGIN) in cell units.

REWARD (kept here because you will need it for the RL stage later)
  "sparse": +1 when the episode succeeds, else 0.
  "shaped": potential-based progress reward + small penalties + success bonus.
"""
from collections import deque

import numpy as np

# ------------------------------------------------------------------ constants
G = 12                       # table is G x G cells
MARGIN = 2                   # extra cells to the left of the table that show the robot base
DT = 0.1                     # seconds per control step (time_s = t * DT)
MAX_STEPS = 60
COLORS = ["red", "green", "blue", "yellow"]
COLOR_RGB = {"red": (220, 50, 50), "green": (40, 160, 70), "blue": (50, 100, 220), "yellow": (240, 200, 40)}
TABLE_RGB, FLOOR_RGB, OBST_RGB = (239, 230, 210), (200, 200, 205), (80, 80, 80)
NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
INF = 10 ** 6


def bfs(obst, sources):
    """Moves needed from every cell to the nearest source cell (8-connected, avoiding obstacles)."""
    dist = np.full((G, G), INF, dtype=np.int64)
    q = deque()
    for (x, y) in sources:
        dist[x, y] = 0
        q.append((x, y))
    while q:
        x, y = q.popleft()
        d = dist[x, y] + 1
        for dx, dy in NB8:
            nx, ny = x + dx, y + dy
            if 0 <= nx < G and 0 <= ny < G and not obst[nx, ny] and dist[nx, ny] > d:
                dist[nx, ny] = d
                q.append((nx, ny))
    return dist


class PickPlaceEnv:
    def __init__(self, reward_mode="shaped", px=6, max_steps=MAX_STEPS, gamma=0.99, seed=None):
        assert reward_mode in ("shaped", "sparse")
        self.reward_mode, self.px, self.max_steps, self.gamma = reward_mode, px, max_steps, gamma
        self.rng = np.random.default_rng(seed)

    # ============================================================== reset / scene generation
    def reset(self, seed=None):
        """Start a new random scene. Passing a seed makes the scene reproducible."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        while not self._sample_layout():
            pass
        self.t, self.gripper_closed, self.held, self._cache = 0, False, None, {}
        return self.observe()

    def _sample_layout(self):
        rng = self.rng
        obst = np.zeros((G, G), bool)                         # two wall segments
        for _ in range(2):
            ln = int(rng.integers(2, 5))
            if rng.random() < 0.5:
                x, y = int(rng.integers(0, G)), int(rng.integers(0, G - ln + 1)); obst[x, y:y + ln] = True
            else:
                x, y = int(rng.integers(0, G - ln + 1)), int(rng.integers(0, G)); obst[x:x + ln, y] = True
        rc = rng.permutation(len(COLORS))[:2]                  # container + zone, distinct colours
        cen = [(int(rng.integers(1, G - 1)), int(rng.integers(1, G - 1))) for _ in range(2)]
        if max(abs(cen[0][0] - cen[1][0]), abs(cen[0][1] - cen[1][1])) < 4:
            return False
        regions = [dict(kind="container", color=int(rc[0]), cx=cen[0][0], cy=cen[0][1]),
                   dict(kind="zone", color=int(rc[1]), cx=cen[1][0], cy=cen[1][1])]
        masks = []
        for r in regions:
            m = np.zeros((G, G), bool); m[r["cx"] - 1:r["cx"] + 2, r["cy"] - 1:r["cy"] + 2] = True; masks.append(m)
        if (obst & (masks[0] | masks[1])).any():
            return False
        free = [(x, y) for x in range(G) for y in range(G) if not obst[x, y] and not masks[0][x, y] and not masks[1][x, y]]
        idx = rng.permutation(len(free))[:3]
        bcols = rng.permutation(len(COLORS))[:3]
        blocks = [dict(color=int(bcols[i]), pos=free[idx[i]]) for i in range(3)]
        cells = [(x, y) for x in range(G) for y in range(G) if not obst[x, y]]
        ee = cells[int(rng.integers(len(cells)))]
        tb, tr = int(rng.integers(3)), int(rng.integers(2))
        d = bfs(obst, [blocks[tb]["pos"]])                     # make sure the task is solvable
        if d[ee] >= INF or d[regions[tr]["cx"], regions[tr]["cy"]] >= INF:
            return False
        self.obst, self.regions, self.region_masks = obst, regions, masks
        self.blocks, self.ee, self.target_block, self.target_region = blocks, ee, tb, tr
        r = regions[tr]
        self.instruction = (f"put the {COLORS[blocks[tb]['color']]} block "
                            f"{'in' if r['kind'] == 'container' else 'on'} the {COLORS[r['color']]} {r['kind']}")
        return True

    # ============================================================== observation (what the policy sees)
    def observe(self):
        return dict(image=self.render(), instruction=self.instruction, t=self.t, time_s=round(self.t * DT, 3))

    def render(self):
        """RGB picture of the scene, uint8 array of shape (G*px, (G+MARGIN)*px, 3)."""
        px = self.px
        cell = np.zeros((G, G + MARGIN, 3), np.float32)
        cell[:, :MARGIN] = FLOOR_RGB
        cell[:, MARGIN:] = TABLE_RGB
        for r, m in zip(self.regions, self.region_masks):      # container = strong tint, zone = pale tint
            a = 0.55 if r["kind"] == "container" else 0.22
            col = np.array(COLOR_RGB[COLORS[r["color"]]], np.float32)
            xs, ys = np.nonzero(m)
            cell[G - 1 - ys, xs + MARGIN] = (1 - a) * np.array(TABLE_RGB, np.float32) + a * col
        xs, ys = np.nonzero(self.obst)
        cell[G - 1 - ys, xs + MARGIN] = OBST_RGB
        img = np.repeat(np.repeat(cell, px, 0), px, 1)
        rowc = lambda y: (G - 1 - y) * px                      # top row (in pixels) of cell row for y
        colc = lambda x: (x + MARGIN) * px
        for r in self.regions:                                 # container gets a solid 1-px border
            if r["kind"] == "container":
                c = COLOR_RGB[COLORS[r["color"]]]
                r0, r1 = rowc(r["cy"] + 1), rowc(r["cy"] - 1) + px
                c0, c1 = colc(r["cx"] - 1), colc(r["cx"] + 1) + px
                img[r0:r1, c0:c0 + 1] = c; img[r0:r1, c1 - 1:c1] = c
                img[r0:r0 + 1, c0:c1] = c; img[r1 - 1:r1, c0:c1] = c
        # robot base + arm (a line from the base to the EE)
        mid = (G // 2) * px
        img[mid - 2 * px:mid + 2 * px, 3:MARGIN * px - 3] = (60, 60, 85)
        p0 = np.array([mid, MARGIN * px - 3], float)
        p1 = np.array([rowc(self.ee[1]) + px / 2, colc(self.ee[0]) + px / 2], float)
        for s in np.linspace(0, 1, int(np.abs(p1 - p0).max() * 2) + 2):
            r, c = (p0 + s * (p1 - p0)).astype(int)
            img[r:r + 2, c:c + 2] = (95, 95, 120)
        for b in self.blocks:                                  # blocks: inset coloured squares
            r0, c0 = rowc(b["pos"][1]), colc(b["pos"][0])
            img[r0 + 1:r0 + px - 1, c0 + 1:c0 + px - 1] = COLOR_RGB[COLORS[b["color"]]]
        r0, c0 = rowc(self.ee[1]), colc(self.ee[0])            # EE: black outline; filled centre = closed
        img[r0:r0 + px, c0:c0 + 1] = 0; img[r0:r0 + px, c0 + px - 1:c0 + px] = 0
        img[r0:r0 + 1, c0:c0 + px] = 0; img[r0 + px - 1:r0 + px, c0:c0 + px] = 0
        if self.gripper_closed:
            h = px // 2 - 1
            img[r0 + h:r0 + h + 2, c0 + h:c0 + h + 2] = 0
        return img.astype(np.uint8)

    # ============================================================== privileged state
    def state(self):
        """Ground-truth state. Use it for supervision / debugging, never as policy input."""
        return dict(
            ee=np.array(self.ee, np.int8),
            gripper_closed=bool(self.gripper_closed),
            holding=-1 if self.held is None else int(self.held),
            block_pos=np.array([b["pos"] for b in self.blocks], np.int8),            # (3, 2)
            block_color=np.array([b["color"] for b in self.blocks], np.int8),        # (3,) index into COLORS
            region_color=np.array([r["color"] for r in self.regions], np.int8),      # [container, zone]
            region_center=np.array([[r["cx"], r["cy"]] for r in self.regions], np.int8),
            obstacle=self.obst.copy(),                                               # (G, G) bool, [x, y]
            target_block=self.target_block, target_region=self.target_region)

    # ============================================================== dynamics + reward
    def _phi(self):
        """Progress potential: reach phase -> -dist(EE, block); carry phase -> 1 - dist(block, region)."""
        ex, ey = self.ee
        tb = self.blocks[self.target_block]
        if self.held == self.target_block:
            cx, cy = self.regions[self.target_region]["cx"], self.regions[self.target_region]["cy"]
            return 1.0 - max(abs(ex - cx), abs(ey - cy)) / G
        if self.region_masks[self.target_region][tb["pos"]]:
            return 1.0
        return -max(abs(ex - tb["pos"][0]), abs(ey - tb["pos"][1])) / G

    def step(self, action):
        dx, dy, g = [int(a) for a in action]
        assert dx in (-1, 0, 1) and dy in (-1, 0, 1) and g in (0, 1), f"bad action {action}"
        phi0 = self._phi()
        ex, ey = self.ee
        nx, ny = min(max(ex + dx, 0), G - 1), min(max(ey + dy, 0), G - 1)
        collided = bool(self.obst[nx, ny])
        if collided:
            nx, ny = ex, ey
        self.ee = (nx, ny)
        if self.held is not None:
            self.blocks[self.held]["pos"] = self.ee
        wrong = False
        if g == 1 and not self.gripper_closed:                 # close -> grasp whatever is under the EE
            self.gripper_closed = True
            for i, b in enumerate(self.blocks):
                if b["pos"] == self.ee:
                    self.held, wrong = i, (i != self.target_block)
                    break
        elif g == 0 and self.gripper_closed:                   # open -> release
            self.gripper_closed, self.held = False, None
        success = self.held is None and bool(self.region_masks[self.target_region][self.blocks[self.target_block]["pos"]])
        self.t += 1
        if self.reward_mode == "sparse":
            r = 1.0 if success else 0.0
        else:
            r = (2.0 * (self.gamma * self._phi() - phi0) - 0.01 - (0.05 if collided else 0.0)
                 - (0.1 if wrong else 0.0) + (5.0 if success else 0.0))
        done = success or self.t >= self.max_steps
        return self.observe(), float(r), done, dict(success=success, collided=collided, wrong_grasp=wrong)

    # ============================================================== scripted expert (the "teleoperator")
    def expert_action(self):
        """Ground-truth action for the CURRENT state, from a shortest-path planner."""
        ex, ey = self.ee
        tb = self.blocks[self.target_block]
        if self.held == self.target_block:
            mask = self.region_masks[self.target_region]
            if mask[ex, ey]:
                return (0, 0, 0)                                # above the region: open gripper
            key, src, grip = ("r", self.target_region), list(zip(*np.nonzero(mask))), 1
        else:
            if self.gripper_closed:
                return (0, 0, 0)                                # closed on nothing / wrong block: re-open
            if self.ee == tb["pos"]:
                return (0, 0, 1)                                # on the target block: close
            key, src, grip = ("b", tb["pos"]), [tb["pos"]], 0
        if key not in self._cache:
            self._cache[key] = bfs(self.obst, src)
        d = self._cache[key]
        best, best_v = (0, 0), INF + 1
        for dx, dy in NB8:
            nx, ny = ex + dx, ey + dy
            if 0 <= nx < G and 0 <= ny < G and not self.obst[nx, ny]:
                v = d[nx, ny] * 3 + abs(dx) + abs(dy)          # prefer shorter, then straighter
                if v < best_v:
                    best, best_v = (dx, dy), v
        return (best[0], best[1], grip)


# ===================================================================== test harness
def evaluate(policy_fn, n_episodes=100, seed0=2_000_000, reward_mode="shaped", px=6):
    """
    Test ANY policy (your future VLA) on fixed scenes.
        policy_fn(image, instruction, t) -> (dx, dy, grip)
    Returns dict(success_rate, mean_steps, mean_return).
    """
    env = PickPlaceEnv(reward_mode, px=px)
    succ, steps, rets = [], [], []
    for i in range(n_episodes):
        obs, ret = env.reset(seed=seed0 + i), 0.0
        while True:
            obs, r, done, info = env.step(policy_fn(obs["image"], obs["instruction"], obs["t"]))
            ret += r
            if done:
                break
        succ.append(info["success"]); steps.append(env.t); rets.append(ret)
    return dict(success_rate=float(np.mean(succ)), mean_steps=float(np.mean(steps)), mean_return=float(np.mean(rets)))
