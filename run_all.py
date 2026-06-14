"""Run the required Kuhn Poker experiments end to end."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")

from task_a1_baseline import run_baseline_experiment
from task_a2_ppo_training import plot_training_curves, train_ppo
from task_a3_exploitability import main as run_exploitability


def main():
    Path("results").mkdir(exist_ok=True)

    np.random.seed(42)
    torch.manual_seed(42)

    player_returns = run_baseline_experiment(num_games=10000)
    baseline_results = {
        "num_games": 10000,
        "player_0_avg": float(np.mean(player_returns[0])),
        "player_0_std": float(np.std(player_returns[0])),
        "player_1_avg": float(np.mean(player_returns[1])),
        "player_1_std": float(np.std(player_returns[1])),
    }
    with open("results/task_a1_baseline.json", "w") as f:
        json.dump(baseline_results, f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agents, history = train_ppo(
        num_episodes=50000,
        update_interval=100,
        eval_interval=1000,
        device=device,
    )
    plot_training_curves(history)

    history_serializable = {
        "episodes": history["episodes"],
        "avg_returns": history["avg_returns"],
        "eval_returns": history["eval_returns"],
        "eval_win_rates": history["eval_win_rates"],
    }
    with open("results/task_a2_history.json", "w") as f:
        json.dump(history_serializable, f, indent=2)

    run_exploitability()


if __name__ == "__main__":
    main()
