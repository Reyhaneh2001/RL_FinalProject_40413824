# environments/generator.py

import random
from collections import deque


class MazeGenerator:
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, size=17, seed=2):
        self.size = size
        self.seed = seed
        self.rng = random.Random(seed)

    def generate(self, wall_ratio=0.18, num_penalties=8, dynamic_feature="energy"):
        while True:
            grid = [["." for _ in range(self.size)] for _ in range(self.size)]

            start = (0, 0)
            goal = (self.size - 1, self.size - 1)
            key = (self.size // 2, self.size // 2)

            grid[start[0]][start[1]] = "S"
            grid[goal[0]][goal[1]] = "G"
            grid[key[0]][key[1]] = "K"

            total_cells = self.size * self.size
            min_walls = max(5, int(total_cells * wall_ratio))

            blocked = {start, goal, key}
            free_cells = [
                (r, c)
                for r in range(self.size)
                for c in range(self.size)
                if (r, c) not in blocked
            ]

            self.rng.shuffle(free_cells)

            walls = free_cells[:min_walls]
            for r, c in walls:
                grid[r][c] = "#"

            remain = [cell for cell in free_cells if cell not in walls]
            self.rng.shuffle(remain)

            penalties = remain[:num_penalties]
            for r, c in penalties:
                grid[r][c] = "P"

            if dynamic_feature == "energy":
                energy_cells = remain[num_penalties:num_penalties + 3]
                for r, c in energy_cells:
                    grid[r][c] = "E"

            if self._is_valid_map(grid, start, key, goal):
                return {
                    "grid": grid,
                    "start": start,
                    "goal": goal,
                    "key": key,
                    "dynamic_feature": dynamic_feature,
                }

    def _is_valid_map(self, grid, start, key, goal):
        return self._bfs(grid, start, key) and self._bfs(grid, key, goal)

    def _bfs(self, grid, src, dst):
        q = deque([src])
        visited = {src}
        while q:
            r, c = q.popleft()
            if (r, c) == dst:
                return True
            for dr, dc in self.ACTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if (nr, nc) not in visited and grid[nr][nc] != "#":
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return False
