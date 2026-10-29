"""
dagger_collect.py -- let a trained VLA drive the simulator, and label every state it visits
with the scripted expert's action (DAgger).

    python dagger_collect.py --policy vla_best.pt --episodes 300 --beta 0.3 --out dagger_r1.npz

Why: behavior cloning only ever sees states the EXPERT visits.  Once the learned policy makes a small
mistake it lands in a state it has never seen and keeps drifting.  Here the learner drives, so the data
covers the states it really gets into, and the expert tells it what it should have done there.

  --beta   probability that the expert's action is EXECUTED instead of the learner's (labels are always
           the expert's).  0.3 for the first round is a good start; 0 for later rounds.
The output file has the same format as every other dataset; pass it to train_vla.py with --extra.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

import pickplace_env as E
from make_dataset import save_arrays
from train_vla import get_device, load_ckpt, model_forward
from vec_env import collect_dagger


def main(a):
    device = get_device(a.device)
    model, cfg = load_ckpt(a.policy, device)
    model.eval()
    use_time = cfg["use_time"]

    @torch.no_grad()
    def policy(images, tokens, t):
        img = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float() / 255.0
        out = model_forward(model, img, torch.from_numpy(tokens).to(device),
                            torch.from_numpy(t).to(device) / E.MAX_STEPS, use_time)
        picks = [torch.multinomial(F.softmax(out[nm], -1), 1).squeeze(1) for nm in ("dx", "dy", "grip")]
        c = torch.stack(picks, 1).cpu().numpy()
        return np.stack([c[:, 0] - 1, c[:, 1] - 1, c[:, 2]], 1)

    t0 = time.time()
    writer, st = collect_dagger(policy, a.episodes, a.beta, a.seed0, a.parallel, a.seed)
    save_arrays(writer.to_arrays(), a.out)
    print(f"saved {st['rows']} labelled states from {st['episodes']} episodes -> {a.out}  [{time.time() - t0:.0f}s]")
    print(f"learner success while collecting (beta={a.beta}): {st['success_rate']:.2f}   mean episode length {st['mean_len']:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="checkpoint (.pt) of the model that will drive")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--seed0", type=int, default=8_000_000, help="first scene seed (use a new range each round)")
    ap.add_argument("--parallel", type=int, default=32)
    ap.add_argument("--out", default="dagger_r1.npz")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
