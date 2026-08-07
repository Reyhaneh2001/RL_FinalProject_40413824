# agents/value_iteration.py

import time
import numpy as np


class ValueIterationAgent:
    def __init__(
        self,
        env,
        gamma=0.95,
        theta=1e-6,
        max_iterations=1000,
    ):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations

        self.n_actions = env.action_space
        self.states = env.get_all_states()
        self.state_to_id = {s: env.state_to_index(s) for s in self.states}

        self.V = np.zeros(env.state_space, dtype=np.float64)
        self.policy = np.zeros(env.state_space, dtype=np.int32)

        self.training_stats = {
            "iterations": 0,
            "deltas": [],
            "runtime_sec": 0.0,
        }

    def q_value(self, state, action):
        q = 0.0
        transitions = self.env.transition_from_state(state, action)
        for prob, next_state, reward, done, _ in transitions:
            next_idx = self.env.state_to_index(next_state)
            q += prob * (reward + (0.0 if done else self.gamma * self.V[next_idx]))
        return q

    def train(self):
        start_time = time.time()

        for iteration in range(1, self.max_iterations + 1):
            delta = 0.0
            new_V = self.V.copy()

            for state in self.states:
                idx = self.env.state_to_index(state)

                if state[0:2] == self.env.goal and state[2] == 1:
                    continue

                q_values = np.array(
                    [self.q_value(state, action) for action in range(self.n_actions)],
                    dtype=np.float64,
                )

                best_v = np.max(q_values)
                delta = max(delta, abs(best_v - self.V[idx]))
                new_V[idx] = best_v

            self.V = new_V
            self.training_stats["deltas"].append(delta)

            if delta < self.theta:
                self.training_stats["iterations"] = iteration
                break
        else:
            self.training_stats["iterations"] = self.max_iterations

        self._extract_policy()
        self.training_stats["runtime_sec"] = time.time() - start_time
        return self.V, self.policy

    def _extract_policy(self):
        for state in self.states:
            idx = self.env.state_to_index(state)
            q_values = np.array(
                [self.q_value(state, action) for action in range(self.n_actions)],
                dtype=np.float64,
            )
            self.policy[idx] = int(np.argmax(q_values))

    def act(self, state):
        idx = self.env.state_to_index(state)
        return int(self.policy[idx])

    def get_state_values_grid(self, has_key=0):
        grid = np.full((self.env.rows, self.env.cols), np.nan, dtype=np.float64)
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                if self.env.original_grid[r][c] != "#":
                    idx = self.env.state_to_index((r, c, has_key))
                    grid[r, c] = self.V[idx]
        return grid

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
                action = self.act(state)
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
