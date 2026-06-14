"""任务 C: 对手策略固定/变化时 PPO agent 的表现差异."""
import json
from pathlib import Path

import numpy as np
import pyspiel
import torch
from tqdm import tqdm

from agents.ppo_agent import PPOAgent


def _pick(action, legal_actions):
    """Return action if legal, otherwise fall back to the first legal action."""
    return action if action in legal_actions else legal_actions[0]


def random_policy(state, player_id, legal_actions):
    """Uniform random opponent."""
    return int(np.random.choice(legal_actions))


def passive_policy(state, player_id, legal_actions):
    """Always pass/check/fold."""
    return _pick(0, legal_actions)


def aggressive_policy(state, player_id, legal_actions):
    """Always bet/call."""
    return _pick(1, legal_actions)


def tight_card_policy(state, player_id, legal_actions):
    """Simple card-aware policy: strong hands bet/call, weak hands pass/fold."""
    info = state.information_state_string(player_id)
    card = int(info[0])
    facing_bet = info.endswith("b")

    if card == 2:
        return _pick(1, legal_actions)
    if card == 0:
        return _pick(0, legal_actions)
    return _pick(1 if facing_bet else 0, legal_actions)


FIXED_POLICIES = {
    "random": random_policy,
    "passive": passive_policy,
    "aggressive": aggressive_policy,
    "tight_card": tight_card_policy,
}


def switching_policy(state, player_id, legal_actions):
    """Opponent changes style from hand to hand."""
    policy = np.random.choice(list(FIXED_POLICIES.values()))
    return policy(state, player_id, legal_actions)


def evaluate_against_policy(game, agent, player_id, opponent_policy, num_games=5000):
    """Evaluate one trained agent against a fixed or switching opponent policy."""
    total_return = 0.0
    wins = 0
    draws = 0

    for _ in tqdm(range(num_games), desc=f"Player {player_id} vs policy", leave=False):
        state = game.new_initial_state()

        while not state.is_terminal():
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                action = np.random.choice(outcomes, p=probs)
                state.apply_action(action)
                continue

            current_player = state.current_player()
            legal_actions = state.legal_actions()

            if current_player == player_id:
                info_state = state.information_state_tensor(current_player)
                action, _ = agent.select_action(info_state, legal_actions, training=False)
            else:
                action = opponent_policy(state, current_player, legal_actions)

            state.apply_action(action)

        player_return = state.returns()[player_id]
        total_return += player_return
        wins += int(player_return > 0)
        draws += int(player_return == 0)

    return {
        "avg_return": total_return / num_games,
        "win_rate": wins / num_games,
        "draw_rate": draws / num_games,
    }


def load_agents(game):
    """Load trained PPO agents from task A2."""
    state_dim = game.information_state_tensor_shape()[0]
    action_dim = game.num_distinct_actions()
    agents = [
        PPOAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=128, device="cpu")
        for _ in range(game.num_players())
    ]

    for player_id, agent in enumerate(agents):
        agent.load(f"results/models/ppo_player_{player_id}.pt")

    return agents


def main(num_games=5000):
    np.random.seed(42)
    torch.manual_seed(42)

    game = pyspiel.load_game("kuhn_poker")
    agents = load_agents(game)

    policies = dict(FIXED_POLICIES)
    policies["switching"] = switching_policy

    results = {}
    print("=" * 60)
    print("任务 C: 对手策略固定/变化实验")
    print("=" * 60)

    for policy_name, policy_fn in policies.items():
        results[policy_name] = {}
        print(f"\n对手策略: {policy_name}")

        for player_id, agent in enumerate(agents):
            metrics = evaluate_against_policy(
                game, agent, player_id, policy_fn, num_games=num_games
            )
            results[policy_name][f"player_{player_id}"] = metrics
            print(
                f"  Player {player_id}: "
                f"avg_return={metrics['avg_return']:.4f}, "
                f"win_rate={metrics['win_rate']:.2%}"
            )

    Path("results").mkdir(exist_ok=True)
    with open("results/task_c_opponent_modeling.json", "w") as f:
        json.dump({"num_games": num_games, "results": results}, f, indent=2)

    print("\n结果已保存到 results/task_c_opponent_modeling.json")


if __name__ == "__main__":
    main()
