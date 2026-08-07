# tests/test_env.py

import numpy as np

from environments.maze import DynamicMazeEnv


def tiny_env():
    grid = [
        ["S", "K", "G", "."],
        [".", ".", ".", "."],
        [".", "#", ".", "."],
        [".", ".", ".", "."],
    ]
    start = (0, 0)
    key = (0, 1)
    goal = (0, 2)
    return DynamicMazeEnv(
        grid=grid, start=start, goal=goal, key=key, reward_mode="shaped",
        step_success_prob=1.0, energy_max=50
    )


def test_wall_bounce_and_bounds():
    env = tiny_env()
    # Move down into a wall neighbor setup
    env.agent_pos = (2, 0)  # left of a wall at (2,1)
    s = env.get_state()
    ns, r, done, info = env.step(action=3)  # right into wall
    assert ns[0] == s[0] and ns[1] == s[1]
    assert info["hit_wall"]


def test_reward_modes_diff():
    # Shaped has move_cost, sparse doesn't for normal move
    grid = [
        ["S", ".", "."],
        [".", ".", "."],
        [".", ".", "G"],
    ]
    start, key, goal = (0, 0), (0, 1), (2, 2)
    env_shaped = DynamicMazeEnv(grid, start, goal, key, reward_mode="shaped", step_success_prob=1.0)
    env_sparse = DynamicMazeEnv(grid, start, goal, key, reward_mode="sparse", step_success_prob=1.0)
    s = env_shaped.reset()
    _, r_shaped, _, _ = env_shaped.step(3)  # right
    env_sparse.reset()
    _, r_sparse, _, _ = env_sparse.step(3)
    assert r_shaped < r_sparse  # shaped includes move_cost negative
