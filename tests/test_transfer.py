# tests/test_transfer.py

import numpy as np

from agents.q_learning import QLearningAgent
from environments.maze import DynamicMazeEnv
from transfer.transfer_learning import create_destination_env, transfer_q_table


def simple_env():
    grid = [
        ["S", ".", ".", "G"],
        [".", ".", ".", "."],
        [".", "K", ".", "."],
        [".", ".", ".", "."],
    ]
    start, goal, key = (0, 0), (0, 3), (2, 1)
    env = DynamicMazeEnv(grid, start, goal, key, reward_mode="shaped", step_success_prob=1.0, energy_max=40)
    return env


def test_transfer_scaling():
    env = simple_env()
    agent = QLearningAgent(env, episodes=10, max_steps=50)
    agent.train()
    Q = agent.Q
    for beta in [0.25, 0.5, 0.75, 1.0]:
        Qb = transfer_q_table(Q, beta=beta)
        assert np.allclose(Qb, Q * beta)


def test_create_destination_env_changes_cells():
    env = simple_env()
    new_grid, start, goal, key = create_destination_env(env, change_ratio=0.2, move_key=True, move_goal=True)
    # Ensure start preserved
    assert start == env.start
    # Ensure there's at least one difference with original grid (unless unlucky)
    changed = sum(1 for r in range(env.rows) for c in range(env.cols) if env.original_grid[r][c] != new_grid[r][c])
    assert changed >= 1
