"""PPO Agent 实现"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class PolicyNetwork(nn.Module):
    """策略网络"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value


class PPOAgent:
    """PPO Agent for Kuhn Poker"""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        device="cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.action_dim = action_dim

        self.policy = PolicyNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = {
            "states": [],
            "actions": [],
            "action_masks": [],
            "rewards": [],
            "log_probs": [],
            "values": [],
            "dones": [],
        }

    def get_state_representation(self, info_state):
        """将 info state 转换为张量"""
        return torch.FloatTensor(info_state).to(self.device)

    def select_action(self, info_state, legal_actions, training=True):
        """选择动作"""
        if training:
            # 训练模式：需要保留梯度
            state = torch.FloatTensor(info_state).to(self.device)
            logits, value = self.policy(state)

            # 创建合法动作的 mask
            action_mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
            action_mask[legal_actions] = True
            masked_logits = logits.masked_fill(~action_mask, float("-inf"))

            # 采样动作
            dist = Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # 保存到 buffer（保留梯度）
            self.buffer["states"].append(state)
            self.buffer["actions"].append(action)
            self.buffer["action_masks"].append(action_mask)
            self.buffer["log_probs"].append(log_prob)
            self.buffer["values"].append(value)
            self.buffer["rewards"].append(0.0)
            self.buffer["dones"].append(False)

            return action.item(), log_prob.item()
        else:
            # 评估模式：不需要梯度
            with torch.no_grad():
                state = torch.FloatTensor(info_state).to(self.device)
                logits, value = self.policy(state)

                # 创建合法动作的 mask
                action_mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
                action_mask[legal_actions] = True
                masked_logits = logits.masked_fill(~action_mask, float("-inf"))

                # 采样动作
                dist = Categorical(logits=masked_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                return action.item(), log_prob.item()

    def store_final_reward(self, reward):
        """存储最终奖励（替换最后一个奖励）"""
        if len(self.buffer["rewards"]) > 0:
            self.buffer["rewards"][-1] = reward
            self.buffer["dones"][-1] = True

    def update(self):
        """PPO 更新"""
        if len(self.buffer["states"]) == 0:
            return {}

        # 转换为张量
        states = torch.stack(self.buffer["states"])
        actions = torch.stack(self.buffer["actions"])
        action_masks = torch.stack(self.buffer["action_masks"])
        old_log_probs = torch.stack(self.buffer["log_probs"])
        rewards = torch.FloatTensor(self.buffer["rewards"]).to(self.device)
        dones = torch.FloatTensor(self.buffer["dones"]).to(self.device)
        old_values = torch.stack(self.buffer["values"]).view(-1).detach()

        # 计算 returns 和 advantages
        returns = []
        advantages = []
        running_return = 0
        running_advantage = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
                running_advantage = 0

            running_return = rewards[t] + self.gamma * running_return
            returns.insert(0, running_return)

            td_error = rewards[t] + self.gamma * (
                old_values[t + 1] if t + 1 < len(old_values) else 0
            ) * (1 - dones[t]) - old_values[t]
            running_advantage = td_error + self.gamma * 0.95 * running_advantage * (1 - dones[t])
            advantages.insert(0, running_advantage)

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # PPO 更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.k_epochs):
            # 重新计算 logits 和 values
            logits, values = self.policy(states)
            values = values.view(-1)
            masked_logits = logits.masked_fill(~action_masks, float("-inf"))

            # 计算新的 log probs
            dist = Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # 计算 ratio
            ratio = torch.exp(new_log_probs - old_log_probs.detach())

            # 计算 surrogate loss
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages.detach()
            policy_loss = -torch.min(surr1, surr2).mean()

            # 计算 value loss
            value_loss = F.mse_loss(values, returns.detach())

            # 总 loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            # 更新网络
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        # 清空 buffer
        self.buffer = {
            "states": [],
            "actions": [],
            "action_masks": [],
            "rewards": [],
            "log_probs": [],
            "values": [],
            "dones": [],
        }

        return {
            "policy_loss": total_policy_loss / self.k_epochs,
            "value_loss": total_value_loss / self.k_epochs,
            "entropy": total_entropy / self.k_epochs,
        }

    def save(self, path):
        """保存模型"""
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        """加载模型"""
        self.policy.load_state_dict(torch.load(path))
