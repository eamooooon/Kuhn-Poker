"""任务 A1: Kuhn Poker 随机策略 baseline
双方随机策略对打 ≥10000 局，统计先手/后手平均收益
"""
import pyspiel
import numpy as np
from tqdm import tqdm
from pathlib import Path


def play_random_game(game):
    """玩一局随机策略的游戏，返回每个玩家的收益"""
    state = game.new_initial_state()

    while not state.is_terminal():
        if state.is_chance_node():
            # 机会节点：按照游戏规则的概率分布采样
            outcomes, probs = zip(*state.chance_outcomes())
            action = np.random.choice(outcomes, p=probs)
        else:
            # 玩家节点：随机选择合法动作
            legal_actions = state.legal_actions()
            action = np.random.choice(legal_actions)

        state.apply_action(action)

    # 返回每个玩家的收益
    returns = state.returns()
    return returns


def run_baseline_experiment(num_games=10000):
    """运行 baseline 实验"""
    game = pyspiel.load_game("kuhn_poker")

    print(f"游戏: {game.get_type().long_name}")
    print(f"玩家数: {game.num_players()}")
    print(f"运行 {num_games} 局游戏...\n")

    # 记录每个玩家的收益
    player_returns = [[] for _ in range(game.num_players())]

    for _ in tqdm(range(num_games), desc="Playing games"):
        returns = play_random_game(game)
        for player_id, ret in enumerate(returns):
            player_returns[player_id].append(ret)

    # 统计结果
    print("\n" + "="*60)
    print("随机策略 Baseline 结果")
    print("="*60)

    for player_id in range(game.num_players()):
        returns = player_returns[player_id]
        avg_return = np.mean(returns)
        std_return = np.std(returns)

        position = "先手 (Player 0)" if player_id == 0 else "后手 (Player 1)"
        print(f"{position}:")
        print(f"  平均收益: {avg_return:.6f} ± {std_return:.6f}")
        print(f"  总收益: {np.sum(returns):.2f}")
        print(f"  胜率: {np.sum(np.array(returns) > 0) / len(returns) * 100:.2f}%")
        print(f"  平局率: {np.sum(np.array(returns) == 0) / len(returns) * 100:.2f}%")
        print(f"  败率: {np.sum(np.array(returns) < 0) / len(returns) * 100:.2f}%")
        print()

    # 验证零和性质
    total_returns = [sum(player_returns[i]) for i in range(game.num_players())]
    print(f"零和性质验证: {sum(total_returns):.6f} (应接近 0)")
    print("="*60)

    return player_returns


if __name__ == "__main__":
    np.random.seed(42)
    player_returns = run_baseline_experiment(num_games=10000)

    # 保存结果
    import json
    results = {
        "num_games": 10000,
        "player_0_avg": float(np.mean(player_returns[0])),
        "player_0_std": float(np.std(player_returns[0])),
        "player_1_avg": float(np.mean(player_returns[1])),
        "player_1_std": float(np.std(player_returns[1])),
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/task_a1_baseline.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n结果已保存到 results/task_a1_baseline.json")
