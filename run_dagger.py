"""
run_dagger.py -- repeat  [collect with the current model -> retrain on everything so far]  a few times.

    python run_dagger.py --base vla_data_5k.npz --init vla_best.pt --human human_data.npz --rounds 3

It just calls dagger_collect.py and train_vla.py for you (so you can also run those by hand).
Round r writes dagger_r{r}.npz and vla_dagger_r{r}.pt.  The beta (expert-executes probability) shrinks
every round: 0.3, 0.15, 0, 0 ...
"""
import argparse
import subprocess
import sys


def run(cmd):
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(a):
    py, ckpt, extras = sys.executable, a.init, []
    for r in range(1, a.rounds + 1):
        out, beta = f"dagger_r{r}.npz", max(0.3 - 0.15 * (r - 1), 0.0)
        run([py, "dagger_collect.py", "--policy", ckpt, "--episodes", str(a.episodes), "--beta", str(beta),
             "--seed0", str(8_000_000 + r * 100_000), "--out", out])
        extras.append(out)
        new = f"vla_dagger_r{r}.pt"
        cmd = [py, "train_vla.py", "--data", a.base, "--extra", *extras, "--init", ckpt, "--epochs", str(a.epochs),
               "--lr", str(a.lr), "--best-path", new, "--last-path", f"vla_dagger_r{r}_last.pt", "--data-on-cpu"]
        if a.human:
            cmd += ["--human", a.human]
        run(cmd)
        ckpt = new
    print(f"\nfinished. Best model of the last round: {ckpt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="expert dataset, e.g. vla_data_5k.npz")
    ap.add_argument("--init", required=True, help="starting checkpoint, e.g. vla_best.pt")
    ap.add_argument("--human", default=None)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=300, help="learner episodes collected per round")
    ap.add_argument("--epochs", type=int, default=10, help="training epochs per round")
    ap.add_argument("--lr", type=float, default=1e-4)
    main(ap.parse_args())
