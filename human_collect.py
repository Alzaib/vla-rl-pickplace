"""
human_collect.py -- drive the simulator with your keyboard and record human demonstrations
in the SAME format as make_dataset.py (so you can train on it or merge it with scripted data).

    python human_collect.py                      # saves to human_data.npz (appends if it exists)
    python human_collect.py --episodes 50        # stop automatically after 50 saved episodes
    python human_collect.py --out my_demos.npz --overwrite

You see exactly what the VLA would see: the camera image + the instruction. Every key press is
ONE control step (hold a key to repeat). The recorded row is: image at time t, instruction, t,
your action, the ground-truth state, and the reward the env gave back.

KEYS (click the window first)
    arrows / W A S D     move up / left / down / right        (up = +y, left = towards the robot base)
    Q E Z C              diagonals: up-left, up-right, down-left, down-right
    SPACE                toggle gripper (open <-> close). Moves keep the current gripper setting,
                         so close it over a block, carry it, then press SPACE again over the target.
    X                    wait (do nothing for one step, gripper unchanged)
    BACKSPACE            undo your last step
    R                    restart this same scene (discard the current attempt)
    N                    skip to a NEW scene (discard the current attempt)
    ESC                  quit (everything already saved stays saved)

Only SUCCESSFUL episodes are saved (automatically, the moment you succeed). If you run out of
time (60 steps) you can undo, retry with R, or skip with N.
"""
import argparse
import copy
import os

import numpy as np

import pickplace_env as E
from make_dataset import DatasetWriter, load_arrays, merge_arrays, save_arrays

MOVE_KEYS = {"up": (0, 1), "w": (0, 1), "down": (0, -1), "s": (0, -1), "left": (-1, 0), "a": (-1, 0),
             "right": (1, 0), "d": (1, 0), "q": (-1, 1), "e": (1, 1), "z": (-1, -1), "c": (1, -1), "x": (0, 0)}
HELP = ("arrows/WASD move   Q E Z C diagonals   SPACE gripper   X wait\n"
        "BACKSPACE undo   R restart scene   N new scene   ESC quit")


class HumanCollector:
    def __init__(self, out, split="train", seed_start=3_000_000, px=6, target=0, overwrite=False):
        self.out, self.split, self.target, self.px = out, split, target, px
        self.env = E.PickPlaceEnv("shaped", px=px)
        self.writer = DatasetWriter(px, noise=None, source="human")
        self.base = None
        if os.path.exists(out) and not overwrite:
            self.base = load_arrays(out)
            human = self.base["ep_seed"][self.base["ep_source"] == "human"]
            if len(human):
                seed_start = max(seed_start, int(human.max()) + 1)       # never reuse an old scene
            print(f"appending to {out} (already has {len(self.base['ep_length'])} episodes)")
        self.seed = seed_start
        self.fig = self.im = self.title = self.status = None
        self.new_scene(same=True)

    # ------------------------------------------------------------------ scene / episode control
    def new_scene(self, same=False):
        if not same:
            self.seed += 1
        self.obs = self.env.reset(seed=self.seed)
        self.st0, self.rows, self.hist = self.env.state(), [], []
        self.grip, self.mode, self.msg = 0, "play", "New scene. Go!"

    @property
    def n_saved(self):
        return self.writer.n_episodes()

    # ------------------------------------------------------------------ keyboard handling
    def handle_key(self, key):
        """All logic lives here (no GUI needed), so it can be tested headlessly."""
        if key == "escape":
            return "quit"
        if key == "n":
            self.new_scene()
        elif key == "r":
            self.new_scene(same=True)
        elif key == "backspace":
            if self.hist and self.mode in ("play", "fail"):
                self.env, self.grip = self.hist.pop()
                self.rows.pop()
                self.obs, self.mode, self.msg = self.env.observe(), "play", "Undid last step."
        elif self.mode == "play":
            if key == " ":
                action, new_grip = (0, 0, 1 - self.grip), 1 - self.grip
            elif key in MOVE_KEYS:
                action, new_grip = (*MOVE_KEYS[key], self.grip), self.grip
            else:
                return None
            self._step(action, new_grip)
        elif self.mode == "success":
            self.msg = "Saved! Press N for the next scene (or R to redo this one)."
        self.redraw()
        if self.mode == "success" and self.target and self.n_saved >= self.target:
            return "quit"
        return None

    def _step(self, action, new_grip):
        self.hist.append((copy.deepcopy(self.env), self.grip))
        st, obs0 = self.env.state(), self.obs
        self.obs, reward, done, info = self.env.step(action)
        self.rows.append(dict(image=obs0["image"], t=obs0["t"], time_s=obs0["time_s"], action=action,
                              ee_pos=st["ee"], gripper_closed=st["gripper_closed"], holding=st["holding"],
                              block_pos=st["block_pos"], reward=reward, done=done, executed_expert=True))
        self.grip = new_grip
        self.msg = ""
        if info["collided"]:
            self.msg = "Bump! (obstacle)"
        if info["wrong_grasp"]:
            self.msg = "Wrong block! Open the gripper and find the right one."
        if done and info["success"]:
            self.writer.add_episode(self.rows, self.env.instruction, True, self.split, self.seed, self.st0)
            self.save()
            self.mode, self.msg = "success", f"SUCCESS in {len(self.rows)} steps - saved ({self.n_saved} this session). Press N."
        elif done:
            self.mode, self.msg = "fail", "Out of time. BACKSPACE = undo, R = retry this scene, N = new scene."

    def save(self):
        new = self.writer.to_arrays()
        save_arrays(new if self.base is None else merge_arrays(self.base, new), self.out)

    # ------------------------------------------------------------------ drawing
    def redraw(self):
        if self.fig is None:
            return
        self.im.set_data(self.obs["image"])
        self.title.set_text(f'"{self.env.instruction}"')
        e = self.env
        held = "nothing" if e.held is None else f"the {E.COLORS[e.blocks[e.held]['color']]} block"
        self.status.set_text(f"step {e.t}/{e.max_steps}   time {e.t * E.DT:.1f}s   gripper {'CLOSED' if self.grip else 'open'}"
                             f"   holding {held}   saved: {self.n_saved}\n{self.msg}")
        self.fig.canvas.draw_idle()

    def attach(self, fig, ax):
        self.fig = fig
        self.im = ax.imshow(self.obs["image"], interpolation="nearest")
        ax.axis("off")
        self.title = ax.set_title("", fontsize=15, fontweight="bold")
        self.status = fig.text(0.5, 0.075, "", ha="center", va="center", fontsize=11)
        fig.text(0.5, 0.02, HELP, ha="center", va="center", fontsize=8, color="gray")
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.redraw()

    def _on_key(self, event):
        if event.key and self.handle_key(event.key) == "quit":
            import matplotlib.pyplot as plt
            plt.close(self.fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="human_data.npz")
    ap.add_argument("--episodes", type=int, default=0, help="stop after this many saved episodes (0 = no limit)")
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--px", type=int, default=6, help="must match your other dataset files if you merge them")
    ap.add_argument("--seed-start", type=int, default=3_000_000)
    ap.add_argument("--overwrite", action="store_true", help="ignore/replace an existing --out file")
    a = ap.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt
    if matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "ps", "cairo", "template"):
        raise SystemExit("matplotlib has no interactive window backend here. Install one, e.g. `pip install pyqt5` "
                         "(or use a Python with tkinter), then run again.")
    for k in list(plt.rcParams):                      # switch off matplotlib's own shortcuts (q = quit, s = save, ...)
        if k.startswith("keymap."):
            plt.rcParams[k] = []
    col = HumanCollector(a.out, a.split, a.seed_start, a.px, a.episodes, a.overwrite)
    fig, ax = plt.subplots(figsize=(8, 7.2))
    fig.subplots_adjust(bottom=0.15)
    fig.canvas.manager.set_window_title("VLA human demo collector")
    col.attach(fig, ax)
    plt.show()
    print(f"\nsession finished: {col.n_saved} new episodes saved in {a.out}")


if __name__ == "__main__":
    main()
