import argparse
import numpy as np
import matplotlib.pyplot as plt

from pickplace_env import PickPlaceEnv


def watch_live():
    env = PickPlaceEnv()

    plt.ion()
    fig, ax = plt.subplots()

    while True:
        obs = env.reset()

        while True:
            ax.clear()
            ax.imshow(obs["image"])
            ax.set_title(
                f"{obs['instruction']}\n"
                f"t={obs['t']}"
            )
            ax.axis("off")

            plt.pause(0.25)

            action = env.expert_action()

            print(
                f"t={obs['t']:2d} "
                f"action={action}"
            )

            obs, reward, done, info = env.step(action)

            if done:
                print("SUCCESS\n")
                plt.pause(1.0)
                break


def replay_dataset(dataset_path, episode):
    d = np.load(dataset_path, allow_pickle=False)

    n_episodes = len(d["ep_length"])

    if episode >= n_episodes:
        raise ValueError(
            f"Episode {episode} does not exist. "
            f"Dataset only contains {n_episodes} episodes."
        )

    start = d["ep_episode_start"][episode]
    length = d["ep_length"][episode]

    print(f"Dataset : {dataset_path}")
    print(f"Episode : {episode}")
    print(f"Steps   : {length}")
    print(f"Task    : {d['ep_instruction'][episode]}")
    print()

    plt.ion()
    fig, ax = plt.subplots()

    for step in range(length):
        idx = start + step

        ax.clear()
        ax.imshow(d["image"][idx])

        action = tuple(int(x) for x in d["action"][idx])

        ax.set_title(
            f"Episode {episode}\n"
            f"{d['ep_instruction'][episode]}\n"
            f"t={d['t'][idx]}  action={action}"
        )

        ax.axis("off")

        print(
            f"t={d['t'][idx]:2d} "
            f"action={action} "
            f"reward={d['reward'][idx]:.2f}"
        )

        plt.pause(0.25)

    plt.ioff()
    plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["live", "replay"],
        default="live",
        help="live = watch expert solve new scenes, replay = replay saved dataset"
    )

    parser.add_argument(
        "--data",
        default="human_data.npz",
        help="Dataset file to replay"
    )

    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to replay"
    )

    args = parser.parse_args()

    if args.mode == "live":
        watch_live()
    else:
        replay_dataset(args.data, args.episode)


if __name__ == "__main__":
    main()