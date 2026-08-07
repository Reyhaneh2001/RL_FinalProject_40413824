# transfer/transfer_learning.py

import copy
import numpy as np


def create_destination_env(source_env, change_ratio=0.2, move_key=False, move_goal=False):
    new_grid = copy.deepcopy(source_env.original_grid)
    rows, cols = source_env.rows, source_env.cols

    protected = {source_env.start, source_env.goal, source_env.key}
    candidates = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (r, c) not in protected and new_grid[r][c] != "#"
    ]

    rng = np.random.default_rng(source_env.seed + 101)
    rng.shuffle(candidates)

    num_changes = max(1, int(len(candidates) * change_ratio))
    chosen = candidates[:num_changes]

    for r, c in chosen:
        if new_grid[r][c] == ".":
            new_grid[r][c] = "P"
        elif new_grid[r][c] == "P":
            new_grid[r][c] = "."
        elif new_grid[r][c] == "E":
            new_grid[r][c] = "."

    key = source_env.key
    goal = source_env.goal

    if move_key:
        empty_cells = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if new_grid[r][c] == "." and (r, c) != source_env.start and (r, c) != goal
        ]
        if empty_cells:
            new_grid[key[0]][key[1]] = "."
            key = empty_cells[int(rng.integers(0, len(empty_cells)))]
            new_grid[key[0]][key[1]] = "K"

    if move_goal:
        empty_cells = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if new_grid[r][c] == "." and (r, c) != source_env.start and (r, c) != key
        ]
        if empty_cells:
            new_grid[goal[0]][goal[1]] = "."
            goal = empty_cells[int(rng.integers(0, len(empty_cells)))]
            new_grid[goal[0]][goal[1]] = "G"

    new_grid[source_env.start[0]][source_env.start[1]] = "S"
    return new_grid, source_env.start, goal, key


def transfer_q_table(source_q, beta=1.0):
    return beta * np.array(source_q, copy=True)


def evaluate_transfer_scenarios(agent_class, source_env, dest_env, train_kwargs, source_q=None):
    scenarios = {}

    zero_agent = agent_class(dest_env, **train_kwargs)
    zero_agent.train()
    scenarios["zero_init"] = {
        "agent": zero_agent,
        "metrics": zero_agent.evaluate_policy(),
    }

    if source_q is not None:
        full_agent = agent_class(dest_env, **train_kwargs)
        full_agent.Q = transfer_q_table(source_q, beta=1.0)
        full_agent.train()
        scenarios["full_transfer"] = {
            "agent": full_agent,
            "metrics": full_agent.evaluate_policy(),
        }

        for beta in [0.25, 0.50, 0.75]:
            beta_agent = agent_class(dest_env, **train_kwargs)
            beta_agent.Q = transfer_q_table(source_q, beta=beta)
            beta_agent.train()
            scenarios[f"scaled_transfer_{beta:.2f}"] = {
                "agent": beta_agent,
                "metrics": beta_agent.evaluate_policy(),
            }

    return scenarios
