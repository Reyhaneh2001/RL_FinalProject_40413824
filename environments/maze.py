# environments/maze.py

import copy
import random


class DynamicMazeEnv:
    ACTIONS = {
        0: (-1, 0),   # up
        1: (1, 0),    # down
        2: (0, -1),   # left
        3: (0, 1),    # right
    }

    PERPENDICULAR = {
        0: [2, 3],
        1: [2, 3],
        2: [0, 1],
        3: [0, 1],
    }

    def __init__(
        self,
        grid,
        start,
        goal,
        key,
        reward_mode="shaped",
        seed=2,
        step_success_prob=0.8,
        penalty_reward=-10.0,
        wall_penalty=-2.0,
        move_cost=-1.0,
        key_reward=20.0,
        goal_reward=100.0,
        energy_max=60,
    ):
        self.original_grid = copy.deepcopy(grid)
        self.grid = copy.deepcopy(grid)
        self.start = start
        self.goal = goal
        self.key = key
        self.reward_mode = reward_mode
        self.seed = seed
        self.rng = random.Random(seed)
        self.step_success_prob = step_success_prob
        self.penalty_reward = penalty_reward
        self.wall_penalty = wall_penalty
        self.move_cost = move_cost
        self.key_reward = key_reward
        self.goal_reward = goal_reward
        self.energy_max = energy_max

        self.rows = len(grid)
        self.cols = len(grid[0])
        self.action_space = 4
        self.state_space = self.rows * self.cols * 2

        self.agent_pos = None
        self.has_key = 0
        self.energy = self.energy_max
        self.done = False
        self.last_info = {}

        self.reset()

    def reset(self):
        self.grid = copy.deepcopy(self.original_grid)
        self.agent_pos = self.start
        self.has_key = 0
        self.energy = self.energy_max
        self.done = False
        self.last_info = {
            "hit_wall": False,
            "collected_key": False,
            "hit_penalty": False,
            "reached_goal": False,
            "energy_depleted": False,
            "recharged": False,
        }
        return self.get_state()

    def get_state(self):
        return (self.agent_pos[0], self.agent_pos[1], self.has_key)

    def state_to_index(self, state):
        r, c, k = state
        return (r * self.cols + c) * 2 + k

    def index_to_state(self, idx):
        cell, k = divmod(idx, 2)
        r, c = divmod(cell, self.cols)
        return (r, c, k)

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_wall(self, r, c):
        return self.grid[r][c] == "#"

    def cell_type(self, pos):
        r, c = pos
        return self.grid[r][c]

    def sample_actual_action(self, intended_action):
        p = self.rng.random()
        if p < self.step_success_prob:
            return intended_action
        elif p < self.step_success_prob + 0.1:
            return self.PERPENDICULAR[intended_action][0]
        else:
            return self.PERPENDICULAR[intended_action][1]

    def transition_from_state(self, state, action):
        r, c, has_key = state
        transitions = []
        probs = [
            (self.step_success_prob, action),
            (0.1, self.PERPENDICULAR[action][0]),
            (0.1, self.PERPENDICULAR[action][1]),
        ]

        for prob, actual_action in probs:
            dr, dc = self.ACTIONS[actual_action]
            nr, nc = r + dr, c + dc

            hit_wall = False
            collected_key = False
            hit_penalty = False
            reached_goal = False
            recharged = False

            if not self.in_bounds(nr, nc) or self.original_grid[nr][nc] == "#":
                nr, nc = r, c
                hit_wall = True

            new_has_key = has_key
            cell = self.original_grid[nr][nc]

            reward = 0.0
            if self.reward_mode == "sparse":
                if cell == "K" and new_has_key == 0:
                    reward += self.key_reward
                    collected_key = True
                    new_has_key = 1
                if (nr, nc) == self.goal and new_has_key == 1:
                    reward += self.goal_reward
                    reached_goal = True
            else:
                reward += self.move_cost
                if hit_wall:
                    reward += self.wall_penalty
                if cell == "P":
                    reward += self.penalty_reward
                    hit_penalty = True
                if cell == "K" and new_has_key == 0:
                    reward += self.key_reward
                    collected_key = True
                    new_has_key = 1
                if cell == "E":
                    reward += 5.0
                    recharged = True
                if (nr, nc) == self.goal and new_has_key == 1:
                    reward += self.goal_reward
                    reached_goal = True

                reward += self._potential_shaping((r, c), (nr, nc), new_has_key)

            done = reached_goal
            next_state = (nr, nc, new_has_key)
            info = {
                "hit_wall": hit_wall,
                "collected_key": collected_key,
                "hit_penalty": hit_penalty,
                "reached_goal": reached_goal,
                "recharged": recharged,
            }
            transitions.append((prob, next_state, reward, done, info))

        return transitions

    def _potential_shaping(self, old_pos, new_pos, has_key):
        target = self.goal if has_key else self.key
        old_d = abs(old_pos[0] - target[0]) + abs(old_pos[1] - target[1])
        new_d = abs(new_pos[0] - target[0]) + abs(new_pos[1] - target[1])
        return 0.3 * (old_d - new_d)

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        self.last_info = {
            "hit_wall": False,
            "collected_key": False,
            "hit_penalty": False,
            "reached_goal": False,
            "energy_depleted": False,
            "recharged": False,
        }

        actual_action = self.sample_actual_action(action)
        r, c = self.agent_pos
        dr, dc = self.ACTIONS[actual_action]
        nr, nc = r + dr, c + dc

        reward = 0.0

        if not self.in_bounds(nr, nc) or self.is_wall(nr, nc):
            nr, nc = r, c
            self.last_info["hit_wall"] = True

        self.agent_pos = (nr, nc)
        cell = self.cell_type(self.agent_pos)

        if self.reward_mode == "shaped":
            reward += self.move_cost
            if self.last_info["hit_wall"]:
                reward += self.wall_penalty

        if cell == "P":
            reward += self.penalty_reward
            self.last_info["hit_penalty"] = True

        if cell == "K" and self.has_key == 0:
            self.has_key = 1
            reward += self.key_reward
            self.last_info["collected_key"] = True

        if cell == "E":
            self.energy = min(self.energy_max, self.energy + 15)
            self.last_info["recharged"] = True
            if self.reward_mode == "shaped":
                reward += 5.0

        if self.reward_mode == "shaped":
            reward += self._potential_shaping((r, c), self.agent_pos, self.has_key)

        self.energy -= 1
        if self.energy <= 0:
            self.done = True
            self.last_info["energy_depleted"] = True
            reward -= 30.0

        if self.agent_pos == self.goal and self.has_key == 1:
            self.done = True
            reward += self.goal_reward
            self.last_info["reached_goal"] = True

        return self.get_state(), reward, self.done, self.last_info.copy()

    def get_all_states(self):
        states = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.original_grid[r][c] != "#":
                    for k in [0, 1]:
                        states.append((r, c, k))
        return states

    def render_ascii(self):
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) == self.agent_pos:
                    row.append("A")
                else:
                    row.append(self.grid[r][c])
            lines.append(" ".join(row))
        return "\n".join(lines)
