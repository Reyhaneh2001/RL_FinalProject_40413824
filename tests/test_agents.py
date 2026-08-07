# tests/test_agents.py

import numpy as np

from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SARSALambdaAgent
from agents.value_iteration import ValueIterationAgent
from environments.maze import DynamicMazeEnv


def small_easy_env():
    grid = [
        ["S", "K", "G"],
        [".", ".", "."],
        [".", ".", "."],
    ]
    start, key, goal = (0, 0), (0, 1), (0, 2)
    env = DynamicMazeEnv(grid, start, goal, key, reward_mode="shaped", step_success_prob=1.0, energy_max=40)
    return env


def test_value_iteration_converges():
    env = small_easy_env()
    vi = ValueIterationAgent(env, gamma=0.95, theta=1e-5, max_iterations=500)
    V, policy = vi.train()
    s_idx = env.state_to_index((0, 0, 0))
    assert np.isfinite(V[s_idx])
    # Greedy action should move right initially toward key
    a = policy[s_idx]
    assert a in (2, 3, 0, 1)  # valid
    # On this tiny env, right is optimal from start
    assert a == 3


def test_q_learning_learns():
    env = small_easy_env()
    agent = QLearningAgent(env, alpha=0.3, gamma=0.95, episodes=200, max_steps=50, epsilon_start=0.8, epsilon_min=0.05)
    agent.train()
    m = agent.evaluate_policy(episodes=20, max_steps=50)
    assert m["success_rate"] >= 0.5


def test_sarsa_lambda_learns():
    env = small_easy_env()
    agent = SARSALambdaAgent(env, alpha=0.3, gamma=0.95, lam=0.8, episodes=200, max_steps=50, epsilon_start=0.8, epsilon_min=0.05)
    agent.train()
    m = agent.evaluate_policy(episodes=20, max_steps=50)
    assert m["success_rate"] >= 0.5
