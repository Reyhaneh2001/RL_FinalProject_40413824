# tests/test_generator.py

import numpy as np

from environments.generator import MazeGenerator


def has_path(grid, src, dst):
    # Simple BFS
    from collections import deque

    rows, cols = len(grid), len(grid[0])
    q = deque([src])
    seen = {src}
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        r, c = q.popleft()
        if (r, c) == dst:
            return True
        for dr, dc in actions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen:
                if grid[nr][nc] != "#":
                    seen.add((nr, nc))
                    q.append((nr, nc))
    return False


def test_generator_bfs_and_ratios():
    gen = MazeGenerator(size=17, seed=123)
    m = gen.generate(wall_ratio=0.18, num_penalties=8)
    grid = m["grid"]
    start, key, goal = m["start"], m["key"], m["goal"]

    assert has_path(grid, start, key)
    assert has_path(grid, key, goal)

    # Wall coverage >= 15%
    grid_arr = np.array(grid)
    wall_ratio = np.mean(grid_arr == "#")
    assert wall_ratio >= 0.15

    # At least 5 obstacles 'P'
    penalties = int(np.sum(grid_arr == "P"))
    assert penalties >= 5
