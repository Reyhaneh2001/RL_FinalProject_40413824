# experiments/analysis.py

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_metrics(result_dir: Path):
    metrics = {}
    for f in result_dir.glob("metrics_*_*.json"):
        algo = f.stem.split("_")[1]
        mode = f.stem.split("_")[-1]
        metrics[(algo, mode)] = json.loads(f.read_text())
    return metrics


def plot_learning_curves(result_dir: Path, out_name="learning_curves.png"):
    dfs = []
    for f in result_dir.glob("trainstats_*_*.csv"):
        algo = f.stem.split("_")[1]
        mode = f.stem.split("_")[-1]
        df = pd.read_csv(f)
        df["algo"] = algo
        df["mode"] = mode
        dfs.append(df)
    if not dfs:
        print("No training CSVs found.")
        return
    df = pd.concat(dfs, ignore_index=True)

    plt.figure(figsize=(10, 6))
    for (algo, mode), group in df.groupby(["algo", "mode"]):
        r = group["reward"].rolling(50, min_periods=1).mean()
        plt.plot(group["episode"], r, label=f"{algo} ({mode})")
    plt.xlabel("Episode")
    plt.ylabel("Reward (rolling mean)")
    plt.title("Learning Curves")
    plt.legend()
    plt.tight_layout()
    out_path = result_dir / out_name
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved {out_path.as_posix()}")


def plot_success_bars(metrics: dict, out_path: Path):
    labels = []
    values = []
    for (algo, mode), m in sorted(metrics.items()):
        labels.append(f"{algo}\n{mode}")
        values.append(m.get("success_rate", 0.0) * 100.0)

    if not labels:
        print("No metrics to plot.")
        return

    plt.figure(figsize=(10, 6))
    xs = np.arange(len(labels))
    plt.bar(xs, values, color="#4C96D7")
    plt.xticks(xs, labels)
    plt.ylabel("Success Rate (%)")
    plt.title("Policy Evaluation Success Rates")
    plt.ylim(0, 100)
    plt.tight_layout()
    out_file = out_path / "success_rates.png"
    plt.savefig(out_file, dpi=180)
    plt.close()
    print(f"Saved {out_file.as_posix()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=str, help="Path to a results/exp_* directory")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        raise SystemExit(f"Dir not found: {result_dir}")

    metrics = load_metrics(result_dir)
    plot_success_bars(metrics, result_dir)
    plot_learning_curves(result_dir)


if __name__ == "__main__":
    main()
