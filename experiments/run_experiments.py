# experiments/run_experiments.py

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SARSALambdaAgent
from agents.value_iteration import ValueIterationAgent
from environments.generator import MazeGenerator
from environments.maze import DynamicMazeEnv
from transfer.transfer_learning import create_destination_env, evaluate_transfer_scenarios


def build_env(
    size,
    seed,
    reward_mode,
    wall_ratio=0.18,
    num_penalties=8,
    success_prob=0.8,
):
    gen = MazeGenerator(size=size, seed=seed)
    m = gen.generate(
        wall_ratio=wall_ratio,
        num_penalties=num_penalties,
        dynamic_feature="energy",
    )
    env = DynamicMazeEnv(
        grid=m["grid"],
        start=m["start"],
        goal=m["goal"],
        key=m["key"],
        reward_mode=reward_mode,
        seed=seed,
        step_success_prob=success_prob,
    )
    return env, m


def plot_grid_image(grid, out_path):
    cmap = {
        "#": (0.2, 0.2, 0.2),
        ".": (1.0, 1.0, 1.0),
        "S": (0.0, 0.6, 1.0),
        "G": (0.1, 0.8, 0.2),
        "K": (1.0, 0.85, 0.0),
        "P": (0.9, 0.2, 0.2),
        "E": (0.6, 0.85, 1.0),
    }
    img = np.zeros((len(grid), len(grid[0]), 3), dtype=float)
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            img[r, c] = cmap.get(grid[r][c], (1.0, 1.0, 1.0))

    plt.figure(figsize=(6, 6))
    plt.imshow(img, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_value_policy_heatmaps(agent, env, out_dir, tag):
    if hasattr(agent, "get_value_grid"):
        v0 = agent.get_value_grid(has_key=0)
        v1 = agent.get_value_grid(has_key=1)
    else:
        v0 = agent.get_state_values_grid(has_key=0)
        v1 = agent.get_state_values_grid(has_key=1)

    def do_plot(v, suffix):
        plt.figure(figsize=(6, 6))
        plt.imshow(v, cmap="viridis")
        plt.colorbar()
        plt.title(f"{tag} {suffix}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{tag}_{suffix}.png", dpi=180)
        plt.close()

    do_plot(v0, "hasKey0_values")
    do_plot(v1, "hasKey1_values")

    if hasattr(agent, "get_policy_grid"):
        p0 = agent.get_policy_grid(has_key=0)
        p1 = agent.get_policy_grid(has_key=1)

        np.save(out_dir / f"{tag}_policy_hasKey0.npy", p0)
        np.save(out_dir / f"{tag}_policy_hasKey1.npy", p1)

        act_cmap = {
            0: (0.9, 0.1, 0.1),
            1: (0.1, 0.9, 0.1),
            2: (0.1, 0.1, 0.9),
            3: (0.9, 0.9, 0.1),
            -1: (0.5, 0.5, 0.5),
        }

        for arr, suffix in [(p0, "hasKey0_policy"), (p1, "hasKey1_policy")]:
            img = np.zeros((env.rows, env.cols, 3))
            for r in range(env.rows):
                for c in range(env.cols):
                    a = int(arr[r, c]) if env.original_grid[r][c] != "#" else -1
                    img[r, c] = act_cmap.get(a, (0.5, 0.5, 0.5))

            plt.figure(figsize=(6, 6))
            plt.imshow(img, interpolation="nearest")
            plt.axis("off")
            plt.title(f"{tag} {suffix}")
            plt.tight_layout()
            plt.savefig(out_dir / f"{tag}_{suffix}.png", dpi=180)
            plt.close()

    np.save(out_dir / f"{tag}_values_hasKey0.npy", v0)
    np.save(out_dir / f"{tag}_values_hasKey1.npy", v1)


def run_experiment(
    student_id="40413824",
    seed=2,
    size=17,
    reward_modes=("shaped", "sparse"),
    algos=("value_iteration", "q_learning", "sarsa_lambda"),
    episodes=3000,
    max_steps=500,
    do_transfer=True,
    wall_ratio=0.18,
    num_penalties=8,
    success_prob=0.8,
    transfer_change_ratio=0.2,
    transfer_move_key=True,
    transfer_move_goal=True,
):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path("results") / f"exp_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    config = {
        "student_id": student_id,
        "seed": seed,
        "size": size,
        "reward_modes": list(reward_modes),
        "algos": list(algos),
        "episodes": episodes,
        "max_steps": max_steps,
        "do_transfer": do_transfer,
        "wall_ratio": wall_ratio,
        "num_penalties": num_penalties,
        "success_prob": success_prob,
        "transfer_change_ratio": transfer_change_ratio,
        "transfer_move_key": transfer_move_key,
        "transfer_move_goal": transfer_move_goal,
    }
    (out_root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for reward in reward_modes:
        env, m = build_env(
            size=size,
            seed=seed,
            reward_mode=reward,
            wall_ratio=wall_ratio,
            num_penalties=num_penalties,
            success_prob=success_prob,
        )

        (out_root / f"grid_{reward}.txt").write_text(
            "\n".join("".join(row) for row in m["grid"]),
            encoding="utf-8",
        )
        plot_grid_image(m["grid"], out_root / f"grid_{reward}.png")

        if "value_iteration" in algos:
            print("Doing experiment on Value Iteration:")
            vi = ValueIterationAgent(env)
            vi.train()
            metrics = vi.evaluate_policy(episodes=50, max_steps=max_steps)
            (out_root / f"metrics_value_iteration_{reward}.json").write_text(
                json.dumps(metrics, indent=2),
                encoding="utf-8",
            )
            save_value_policy_heatmaps(vi, env, out_root, tag=f"vi_{reward}")

        if "q_learning" in algos:
            print("Doing experiment on Q_learning:")
            ql = QLearningAgent(env, episodes=episodes, max_steps=max_steps)
            ql.train()
            metrics = ql.evaluate_policy(episodes=50, max_steps=max_steps)
            (out_root / f"metrics_q_learning_{reward}.json").write_text(
                json.dumps(metrics, indent=2),
                encoding="utf-8",
            )

            df = pd.DataFrame(
                {
                    "episode": np.arange(len(ql.training_stats["episode_rewards"])),
                    "reward": ql.training_stats["episode_rewards"],
                    "length": ql.training_stats["episode_lengths"],
                    "epsilon": ql.training_stats["epsilons"],
                    "success": ql.training_stats["success_flags"],
                    "wall_hits": ql.training_stats["wall_hits"],
                    "penalty_hits": ql.training_stats["penalty_hits"],
                    "key_collections": ql.training_stats["key_collections"],
                    "goal_reaches": ql.training_stats["goal_reaches"],
                }
            )
            df.to_csv(out_root / f"trainstats_q_learning_{reward}.csv", index=False)

            np.save(out_root / f"qtable_{reward}.npy", ql.Q)
            save_value_policy_heatmaps(ql, env, out_root, tag=f"ql_{reward}")

        if "sarsa_lambda" in algos:
            print("Doing experiment on Sarsa Lambda:")
            sa = SARSALambdaAgent(env, episodes=episodes, max_steps=max_steps)
            sa.train()
            metrics = sa.evaluate_policy(episodes=50, max_steps=max_steps)
            (out_root / f"metrics_sarsa_lambda_{reward}.json").write_text(
                json.dumps(metrics, indent=2),
                encoding="utf-8",
            )

            df = pd.DataFrame(
                {
                    "episode": np.arange(len(sa.training_stats["episode_rewards"])),
                    "reward": sa.training_stats["episode_rewards"],
                    "length": sa.training_stats["episode_lengths"],
                    "epsilon": sa.training_stats["epsilons"],
                    "success": sa.training_stats["success_flags"],
                    "wall_hits": sa.training_stats["wall_hits"],
                    "penalty_hits": sa.training_stats["penalty_hits"],
                    "key_collections": sa.training_stats["key_collections"],
                    "goal_reaches": sa.training_stats["goal_reaches"],
                }
            )
            df.to_csv(out_root / f"trainstats_sarsa_lambda_{reward}.csv", index=False)
            save_value_policy_heatmaps(sa, env, out_root, tag=f"sa_{reward}")

        if do_transfer and "q_learning" in algos:
            print("Doing experiment on Transfer Q_learning:")
            new_grid, start, goal, key = create_destination_env(
                env,
                change_ratio=transfer_change_ratio,
                move_key=transfer_move_key,
                move_goal=transfer_move_goal,
            )

            dest_env = DynamicMazeEnv(
                grid=new_grid,
                start=start,
                goal=goal,
                key=key,
                reward_mode=reward,
                seed=seed + 101,
                step_success_prob=success_prob,
            )

            source_q = None
            qtable_path = out_root / f"qtable_{reward}.npy"
            if qtable_path.exists():
                source_q = np.load(qtable_path)

            scenarios = evaluate_transfer_scenarios(
                agent_class=QLearningAgent,
                source_env=env,
                dest_env=dest_env,
                train_kwargs={
                    "episodes": episodes // 2,
                    "max_steps": max_steps,
                },
                source_q=source_q,
            )

            out = {k: v["metrics"] for k, v in scenarios.items()}
            (out_root / f"transfer_metrics_{reward}.json").write_text(
                json.dumps(out, indent=2),
                encoding="utf-8",
            )

    print(f"Experiment results are in: {out_root.as_posix()}")


def main():
    parser = argparse.ArgumentParser(description="Run RL experiments for the dynamic maze project.")
    parser.add_argument("--student-id", default="40413824", type=str)
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--size", default=17, type=int)
    parser.add_argument(
        "--reward-modes",
        nargs="+",
        default=["shaped", "sparse"],
        choices=["shaped", "sparse"],
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=["value_iteration", "q_learning", "sarsa_lambda"],
        choices=["value_iteration", "q_learning", "sarsa_lambda"],
    )
    parser.add_argument("--episodes", default=3000, type=int)
    parser.add_argument("--max-steps", default=500, type=int)
    parser.add_argument("--wall-ratio", default=0.18, type=float)
    parser.add_argument("--num-penalties", default=8, type=int)
    parser.add_argument("--success-prob", default=0.8, type=float)
    parser.add_argument("--do-transfer", action="store_true")
    parser.add_argument("--no-transfer", action="store_true")
    parser.add_argument("--transfer-change-ratio", default=0.2, type=float)
    parser.add_argument("--transfer-move-key", action="store_true")
    parser.add_argument("--transfer-move-goal", action="store_true")

    args = parser.parse_args()

    do_transfer = True
    if args.no_transfer:
        do_transfer = False
    elif args.do_transfer:
        do_transfer = True

    run_experiment(
        student_id=args.student_id,
        seed=args.seed,
        size=args.size,
        reward_modes=tuple(args.reward_modes),
        algos=tuple(args.algos),
        episodes=args.episodes,
        max_steps=args.max_steps,
        do_transfer=do_transfer,
        wall_ratio=args.wall_ratio,
        num_penalties=args.num_penalties,
        success_prob=args.success_prob,
        transfer_change_ratio=args.transfer_change_ratio,
        transfer_move_key=args.transfer_move_key,
        transfer_move_goal=args.transfer_move_goal,
    )


if __name__ == "__main__":
    main()
