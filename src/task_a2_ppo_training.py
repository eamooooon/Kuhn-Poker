"""任务 A2: 用 PPO 训练 Kuhn Poker agent
选择自我对弈训练，因为：
1. 对手策略会随着训练改进，提供更有挑战性的对抗
2. 更接近真实博弈场景（双方都在优化策略）
3. 理论上可以收敛到某个均衡点（虽然不保证是纳什均衡）
"""
import pyspiel
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path

from agents.ppo_agent import PPOAgent


def play_episode_self_play(game, agents, training=True):
    """自我对弈一局游戏"""
    state = game.new_initial_state()
    episode_data = {player_id: [] for player_id in range(len(agents))}

    while not state.is_terminal():
        if state.is_chance_node():
            # 机会节点：按照游戏规则采样
            outcomes, probs = zip(*state.chance_outcomes())
            action = np.random.choice(outcomes, p=probs)
            state.apply_action(action)
        else:
            # 玩家节点
            player_id = state.current_player()
            info_state = state.information_state_tensor(player_id)
            legal_actions = state.legal_actions()

            action, log_prob = agents[player_id].select_action(
                info_state, legal_actions, training=training
            )
            state.apply_action(action)

    # 游戏结束，分配奖励
    returns = state.returns()
    for player_id, agent in enumerate(agents):
        if training:
            agent.store_final_reward(returns[player_id])

    return returns


def evaluate_against_random(game, agent, player_id, num_games=1000):
    """评估 agent 对随机策略的表现"""
    wins = 0
    total_return = 0

    for _ in tqdm(range(num_games), desc=f"Evaluating Player {player_id}", leave=False):
        state = game.new_initial_state()

        while not state.is_terminal():
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                action = np.random.choice(outcomes, p=probs)
                state.apply_action(action)
            else:
                current_player = state.current_player()
                legal_actions = state.legal_actions()

                if current_player == player_id:
                    # 使用训练好的 agent
                    info_state = state.information_state_tensor(current_player)
                    action, _ = agent.select_action(
                        info_state, legal_actions, training=False
                    )
                else:
                    # 随机策略
                    action = np.random.choice(legal_actions)

                state.apply_action(action)

        returns = state.returns()
        total_return += returns[player_id]
        if returns[player_id] > 0:
            wins += 1

    avg_return = total_return / num_games
    win_rate = wins / num_games
    return avg_return, win_rate


def train_ppo(
    num_episodes=50000,
    update_interval=100,
    eval_interval=1000,
    device="cpu",
):
    """训练 PPO agent"""
    game = pyspiel.load_game("kuhn_poker")
    state_dim = game.information_state_tensor_shape()[0]
    action_dim = game.num_distinct_actions()

    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"训练设备: {device}\n")

    # 创建两个 agent（自我对弈）
    agents = [
        PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=3e-4,
            device=device,
        )
        for _ in range(2)
    ]

    # 训练记录
    history = {
        "episodes": [],
        "avg_returns": [[], []],
        "policy_losses": [[], []],
        "value_losses": [[], []],
        "eval_returns": [[], []],
        "eval_win_rates": [[], []],
    }

    returns_buffer = [[], []]

    print("开始训练...")
    for episode in tqdm(range(num_episodes), desc="Training"):
        # 玩一局游戏
        returns = play_episode_self_play(game, agents, training=True)

        for player_id in range(2):
            returns_buffer[player_id].append(returns[player_id])

        # 更新策略
        if (episode + 1) % update_interval == 0:
            for player_id, agent in enumerate(agents):
                losses = agent.update()
                if losses:
                    history["policy_losses"][player_id].append(losses["policy_loss"])
                    history["value_losses"][player_id].append(losses["value_loss"])

            # 记录平均回报
            for player_id in range(2):
                avg_return = np.mean(returns_buffer[player_id])
                history["avg_returns"][player_id].append(avg_return)
                returns_buffer[player_id] = []

            history["episodes"].append(episode + 1)

        # 评估
        if (episode + 1) % eval_interval == 0:
            for player_id, agent in enumerate(agents):
                avg_return, win_rate = evaluate_against_random(
                    game, agent, player_id, num_games=1000
                )
                history["eval_returns"][player_id].append(avg_return)
                history["eval_win_rates"][player_id].append(win_rate)

            print(
                f"\n[Episode {episode + 1}] "
                f"Player 0 vs Random: {history['eval_returns'][0][-1]:.4f} (WR: {history['eval_win_rates'][0][-1]:.2%}) | "
                f"Player 1 vs Random: {history['eval_returns'][1][-1]:.4f} (WR: {history['eval_win_rates'][1][-1]:.2%})"
            )

    # 保存模型
    Path("results/models").mkdir(parents=True, exist_ok=True)
    for player_id, agent in enumerate(agents):
        agent.save(f"results/models/ppo_player_{player_id}.pt")

    return agents, history


def plot_training_curves(history):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 平均回报
    ax = axes[0, 0]
    episodes = history["episodes"]
    for player_id in range(2):
        if history["avg_returns"][player_id]:
            ax.plot(
                episodes,
                history["avg_returns"][player_id],
                label=f"Player {player_id}",
                alpha=0.7,
            )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Return (Self-play)")
    ax.set_title("Training Returns (Self-play)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 对随机策略的表现
    ax = axes[0, 1]
    eval_episodes = np.arange(1000, len(history["eval_returns"][0]) * 1000 + 1, 1000)
    for player_id in range(2):
        if history["eval_returns"][player_id]:
            ax.plot(
                eval_episodes,
                history["eval_returns"][player_id],
                label=f"Player {player_id}",
                marker="o",
                alpha=0.7,
            )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Return vs Random")
    ax.set_title("Evaluation vs Random Policy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Policy Loss
    ax = axes[1, 0]
    for player_id in range(2):
        if history["policy_losses"][player_id]:
            ax.plot(
                episodes,
                history["policy_losses"][player_id],
                label=f"Player {player_id}",
                alpha=0.7,
            )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Policy Loss")
    ax.set_title("Policy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Value Loss
    ax = axes[1, 1]
    for player_id in range(2):
        if history["value_losses"][player_id]:
            ax.plot(
                episodes,
                history["value_losses"][player_id],
                label=f"Player {player_id}",
                alpha=0.7,
            )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Value Loss")
    ax.set_title("Value Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/task_a2_training_curves.png", dpi=300)
    print("\n训练曲线已保存到 results/task_a2_training_curves.png")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agents, history = train_ppo(
        num_episodes=50000, update_interval=100, eval_interval=1000, device=device
    )

    # 绘制训练曲线
    plot_training_curves(history)

    # 保存训练历史
    history_serializable = {
        "episodes": history["episodes"],
        "avg_returns": history["avg_returns"],
        "eval_returns": history["eval_returns"],
        "eval_win_rates": history["eval_win_rates"],
    }

    with open("results/task_a2_history.json", "w") as f:
        json.dump(history_serializable, f, indent=2)

    print("\n训练历史已保存到 results/task_a2_history.json")
    print("\n最终评估结果（对随机策略）:")
    for player_id in range(2):
        print(
            f"Player {player_id}: "
            f"平均收益 = {history['eval_returns'][player_id][-1]:.4f}, "
            f"胜率 = {history['eval_win_rates'][player_id][-1]:.2%}"
        )
