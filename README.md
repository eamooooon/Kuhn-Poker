# Kuhn Poker PPO 项目

> 使用 PPO 算法训练 Kuhn Poker AI，并分析其与纳什均衡的距离

## 一键运行

```bash
# 安装依赖
uv sync
# 一条命令运行所有
uv run python run_all.py
```

或使用 pip：
```bash
pip install -r requirements.txt
python run_all.py
```

也可以单独运行各步骤：

```bash
uv run python src/task_a1_baseline.py
uv run python src/task_a2_ppo_training.py
uv run python src/task_a3_exploitability.py
uv run python src/task_c_opponent_modeling.py
```

## 项目结构

```
kuhn-poker-ppo/
├── src/
│   ├── agents/ppo_agent.py           # PPO 核心实现
│   ├── task_a1_baseline.py           # 任务 A1：随机策略
│   ├── task_a2_ppo_training.py       # 任务 A2：PPO 训练
│   ├── task_a3_exploitability.py     # 任务 A3：Exploitability
│   └── task_c_opponent_modeling.py   # 任务 C：对手策略实验
├── results/
│   ├── models/                       # 训练好的模型
│   ├── task_a1_baseline.json         # 实验数据
│   ├── task_a2_history.json          # 训练历史
│   ├── task_a2_training_curves.png   # 训练曲线
│   ├── task_a3_exploitability.json   # Exploitability 数据
│   └── task_c_opponent_modeling.json # 对手策略实验数据
├── run_all.py                        # 一键运行所有必做实验
├── README.md                         # 本文件
├── REPORT.md                         # 最终报告
└── TASK.md                           # 任务需求
```

## 实验结果

### 任务 A1: 随机策略
- 先手: +0.150, 后手: -0.150

### 任务 A2: PPO 训练
- 训练轮数: 50,000 episodes
- 最终对随机策略平均收益: Player 0 = -0.005, Player 1 = +0.097
- 说明: 自我对弈收益仍有明显振荡，但最终策略的 exploitability 优于随机策略

### 任务 A3: Exploitability
| 策略 | Exploitability |
|------|----------------|
| 随机 | 0.458 |
| PPO  | 0.249 |
| CFR  | 0.0001 (最优) |

### 任务 C: 对手策略实验
| 对手策略 | Player 0 平均收益 | Player 1 平均收益 |
|----------|-------------------|-------------------|
| 随机 | +0.024 | +0.131 |
| 保守 | -0.020 | +0.642 |
| 激进 | +0.009 | -0.093 |
| 牌力策略 | -0.024 | -0.047 |
| 每局切换 | +0.016 | +0.090 |

## 关键发现

**PPO 自我对弈能学到强于随机的策略，但仍远离纳什均衡**

这个结果验证了两点：
- PPO 自我对弈在不完美信息博弈中难以收敛
- 50,000 episodes 后 PPO 比随机策略更难被利用，但离 CFR 仍有数量级差距
- CFR 等有理论保证的算法更适合扑克游戏

## 依赖

- Python >= 3.11
- PyTorch >= 2.0.0
- OpenSpiel >= 1.6.15
- NumPy, Matplotlib, tqdm

## License

MIT
