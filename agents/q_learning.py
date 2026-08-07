# agents/q_learning.py

import time
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        env,
        alpha=0.1,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        decay_strategy="exponential",
        episodes=3000,
        max_steps=500,
        seed=2,
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.decay_strategy = decay_strategy
        self.episodes = episodes
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.n_states = env.state_space
        self.n_actions = env.action_space

        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self.visit_counts = np.zeros((self.n_states, self.n_actions), dtype=np.int32)
        self.state_visit_counts = np.zeros(self.n_states, dtype=np.int32)

        self.logs = []
        self.training_stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "success_flags": [],
            "epsilons": [],
            "wall_hits": [],
            "penalty_hits": [],
            "key_collections": [],
            "goal_reaches": [],
            "runtime_sec": 0.0,
        }

    def get_epsilon(self, episode_idx):
        if self.decay_strategy == "linear":
            fraction = episode_idx / max(1, self.episodes - 1)
            eps = self.epsilon_start - fraction * (self.epsilon_start - self.epsilon_min)
            return max(self.epsilon_min, eps)
        eps = self.epsilon_start * (self.epsilon_decay ** episode_idx)
        return max(self.epsilon_min, eps)

    def select_action(self, state, epsilon):
        state_idx = self.env.state_to_index(state)
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[state_idx]))

    def greedy_action(self, state):
        state_idx = self.env.state_to_index(state)
        return int(np.argmax(self.Q[state_idx]))

    def train(self):
        start_time = time.time()

        for ep in range(self.episodes):
            epsilon = self.get_epsilon(ep)
            state = self.env.reset()
            total_reward = 0.0
            wall_hits = 0
            penalty_hits = 0
            key_collections = 0
            goal_reaches = 0

            for step in range(self.max_steps):
                s_idx = self.env.state_to_index(state)
                self.state_visit_counts[s_idx] += 1

                action = self.select_action(state, epsilon)
                next_state, reward, done, info = self.env.step(action)

                ns_idx = self.env.state_to_index(next_state)

                td_target = reward
                if not done:
                    td_target += self.gamma * np.max(self.Q[ns_idx])

                td_error = td_target - self.Q[s_idx, action]
                self.Q[s_idx, action] += self.alpha * td_error

                self.visit_counts[s_idx, action] += 1
                total_reward += reward

                wall_hits += int(info.get("hit_wall", False))
                penalty_hits += int(info.get("hit_penalty", False))
                key_collections += int(info.get("collected_key", False))
                goal_reaches += int(info.get("reached_goal", False))

                self.logs.append(
                    {
                        "episode": ep,
                        "step": step,
                        "state": state,
                        "action": action,
                        "next_state": next_state,
                        "reward": reward,
                        "done": done,
                        "hit_wall": info.get("hit_wall", False),
                        "collected_key": info.get("collected_key", False),
                        "hit_penalty": info.get("hit_penalty", False),
                        "reached_goal": info.get("reached_goal", False),
                        "energy_depleted": info.get("energy_depleted", False),
                        "recharged": info.get("recharged", False),
                        "epsilon": epsilon,
                    }
                )

                state = next_state
                if done:
                    break

            self.training_stats["episode_rewards"].append(total_reward)
            self.training_stats["episode_lengths"].append(step + 1)
            self.training_stats["success_flags"].append(1 if goal_reaches > 0 else 0)
            self.training_stats["epsilons"].append(epsilon)
            self.training_stats["wall_hits"].append(wall_hits)
            self.training_stats["penalty_hits"].append(penalty_hits)
            self.training_stats["key_collections"].append(key_collections)
            self.training_stats["goal_reaches"].append(goal_reaches)

        self.training_stats["runtime_sec"] = time.time() - start_time
        return self.Q

    def act(self, state):
        return self.greedy_action(state)

    def evaluate_policy(self, episodes=30, max_steps=500):
        rewards = []
        success = 0
        steps_used = []

        for _ in range(episodes):
            state = self.env.reset()
            total_reward = 0.0
            done = False
            steps = 0

            while not done and steps < max_steps:
                action = self.greedy_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                steps += 1

            rewards.append(total_reward)
            steps_used.append(steps)
            if self.env.last_info.get("reached_goal", False):
                success += 1

        return {
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "success_rate": success / episodes if episodes else 0.0,
            "avg_steps": float(np.mean(steps_used)) if steps_used else 0.0,
        }

    def get_value_grid(self, has_key=0):
        grid = np.full((self.env.rows, self.env.cols), np.nan, dtype=np.float64)
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                if self.env.original_grid[r][c] != "#":
                    idx = self.env.state_to_index((r, c, has_key))
                    grid[r, c] = np.max(self.Q[idx])
        return grid

    def get_policy_grid(self, has_key=0):
        grid = np.full((self.env.rows, self.env.cols), -1, dtype=np.int32)
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                if self.env.original_grid[r][c] != "#":
                    idx = self.env.state_to_index((r, c, has_key))
                    grid[r, c] = int(np.argmax(self.Q[idx]))
        return grid
