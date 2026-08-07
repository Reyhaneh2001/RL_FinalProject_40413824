# gui/app.py

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pygame

from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SARSALambdaAgent
from agents.value_iteration import ValueIterationAgent
from environments.generator import MazeGenerator
from environments.maze import DynamicMazeEnv
from gui.renderer import MazeRenderer
from transfer.transfer_learning import create_destination_env


def _build_env(config):
    gen = MazeGenerator(size=config["size"], seed=config["seed"])
    m = gen.generate(
        wall_ratio=config.get("wall_ratio", 0.18),
        num_penalties=config.get("num_penalties", 8),
        dynamic_feature="energy",
    )
    env = DynamicMazeEnv(
        grid=m["grid"],
        start=m["start"],
        goal=m["goal"],
        key=m["key"],
        reward_mode=config["reward_mode"],
        seed=config["seed"],
        step_success_prob=config.get("success_prob", 0.8),
        penalty_reward=-10.0,
        wall_penalty=-2.0,
        move_cost=-1.0,
        key_reward=20.0,
        goal_reward=100.0,
        energy_max=60,
    )
    return env, m


@dataclass
class HyperParams:
    gamma: float = 0.95
    alpha: float = 0.1
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    decay_strategy: str = "exponential"
    lam: float = 0.8


class AgentManager:
    def __init__(self, env: DynamicMazeEnv, algo: str, hp: HyperParams):
        self.env = env
        self.algo = algo
        self.hp = hp
        self.agent = None
        self.trained = False
        self._persist_Q = None
        self.episodes_done = 0
        self.total_steps = 0
        self.last_eval = None
        self._init_agent()

    def _init_agent(self):
        if self.algo == "Value Iteration":
            self.agent = ValueIterationAgent(self.env, gamma=self.hp.gamma)
        elif self.algo == "Q-Learning":
            self.agent = QLearningAgent(
                self.env,
                alpha=self.hp.alpha,
                gamma=self.hp.gamma,
                epsilon_start=self.hp.epsilon_start,
                epsilon_min=self.hp.epsilon_min,
                epsilon_decay=self.hp.epsilon_decay,
                decay_strategy=self.hp.decay_strategy,
                episodes=1,
                max_steps=500,
            )
        elif self.algo == "SARSA(λ)":
            self.agent = SARSALambdaAgent(
                self.env,
                alpha=self.hp.alpha,
                gamma=self.hp.gamma,
                lam=self.hp.lam,
                epsilon_start=self.hp.epsilon_start,
                epsilon_min=self.hp.epsilon_min,
                epsilon_decay=self.hp.epsilon_decay,
                decay_strategy=self.hp.decay_strategy,
                episodes=1,
                max_steps=500,
            )
        else:
            raise ValueError(f"Unknown algo {self.algo}")

    def set_algo(self, algo: str):
        self.algo = algo
        self._persist_Q = None
        self.episodes_done = 0
        self.total_steps = 0
        self.trained = False
        self.last_eval = None
        self._init_agent()

    def set_env(self, env: DynamicMazeEnv):
        self.env = env
        self._init_agent()

        if self._persist_Q is not None and hasattr(self.agent, "Q"):
            if getattr(self.agent, "Q", None) is None:
                self._persist_Q = None
            elif self.agent.Q.shape != self._persist_Q.shape:
                self._persist_Q = None

        self.trained = False
        self.episodes_done = 0
        self.total_steps = 0
        self.last_eval = None

    def train_batch(self, episodes=10) -> Tuple[int, float]:
        start = time.time()

        if self.algo == "Value Iteration":
            if not self.trained:
                self.agent.train()
                self.trained = True
                self.episodes_done = 1
                return 1, time.time() - start
            return 0, time.time() - start

        if self.algo == "Q-Learning":
            trainer = QLearningAgent(
                self.env,
                alpha=self.hp.alpha,
                gamma=self.hp.gamma,
                epsilon_start=self.hp.epsilon_start,
                epsilon_min=self.hp.epsilon_min,
                epsilon_decay=self.hp.epsilon_decay,
                decay_strategy=self.hp.decay_strategy,
                episodes=episodes,
                max_steps=500,
            )
        else:
            trainer = SARSALambdaAgent(
                self.env,
                alpha=self.hp.alpha,
                gamma=self.hp.gamma,
                lam=self.hp.lam,
                epsilon_start=self.hp.epsilon_start,
                epsilon_min=self.hp.epsilon_min,
                epsilon_decay=self.hp.epsilon_decay,
                decay_strategy=self.hp.decay_strategy,
                episodes=episodes,
                max_steps=500,
            )

        if self._persist_Q is not None and hasattr(trainer, "Q"):
            trainer.Q = np.array(self._persist_Q, copy=True)

        trainer.train()

        if hasattr(trainer, "Q"):
            self._persist_Q = np.array(trainer.Q, copy=True)

        self.agent = trainer
        self.trained = True
        self.episodes_done += episodes

        return episodes, time.time() - start

    def evaluate(self, episodes=20, max_steps=500):
        self.last_eval = self.agent.evaluate_policy(episodes=episodes, max_steps=max_steps)
        return self.last_eval

    def value_grid(self, has_key=0):
        if hasattr(self.agent, "get_value_grid"):
            return self.agent.get_value_grid(has_key=has_key)
        if hasattr(self.agent, "get_state_values_grid"):
            return self.agent.get_state_values_grid(has_key=has_key)
        return None

    def policy_grid(self, has_key=0):
        if hasattr(self.agent, "get_policy_grid"):
            return self.agent.get_policy_grid(has_key=has_key)

        if isinstance(self.agent, ValueIterationAgent):
            grid = np.full((self.env.rows, self.env.cols), -1, dtype=np.int32)
            for r in range(self.env.rows):
                for c in range(self.env.cols):
                    if self.env.original_grid[r][c] != "#":
                        s = (r, c, has_key)
                        qs = [self.agent.q_value(s, a) for a in range(self.env.action_space)]
                        grid[r, c] = int(np.argmax(qs))
            return grid

        return None

    def greedy_action(self):
        if self.agent is None or not self.trained:
            return None

        state = self.env.get_state()

        if isinstance(self.agent, ValueIterationAgent):
            qs = [self.agent.q_value(state, a) for a in range(self.env.action_space)]
            return int(np.argmax(qs))

        if hasattr(self.agent, "Q"):
            s_idx = self.env.state_to_index(state)
            return int(np.argmax(self.agent.Q[s_idx]))

        return None
        

def run_app():
    pygame.init()

    config = {
        "size": 17,
        "seed": 2,
        "reward_mode": "shaped",
        "wall_ratio": 0.18,
        "num_penalties": 8,
        "success_prob": 0.8,
    }
    env, _ = _build_env(config)

    renderer = MazeRenderer(env, cell_size=28, side_panel=680, margin=10)
    screen = renderer.surface()
    clock = pygame.time.Clock()

    hp = HyperParams()
    algo = "Q-Learning"
    manager = AgentManager(env, algo, hp)

    running = True
    training = False
    playing = False
    overlay_heatmap = True
    overlay_policy = True
    episodes_per_batch = 10
    last_runtime = 0.0
    last_move_time = 0.0
    move_delay = 0.18

    dest_env: Optional[DynamicMazeEnv] = None
    on_destination = False

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    env.reset()

    while running:
        current_env = dest_env if on_destination else env

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    training = not training

                elif event.key == pygame.K_m:
                    playing = not playing

                elif event.key == pygame.K_r:
                    current_env.reset()

                elif event.key == pygame.K_v:
                    config["reward_mode"] = "sparse" if config["reward_mode"] == "shaped" else "shaped"
                    env, _ = _build_env(config)
                    if not on_destination:
                        manager.set_env(env)
                        renderer.set_env(env)
                        env.reset()

                elif event.key == pygame.K_1:
                    algo = "Value Iteration"
                    manager.set_algo(algo)

                elif event.key == pygame.K_2:
                    algo = "Q-Learning"
                    manager.set_algo(algo)

                elif event.key == pygame.K_3:
                    algo = "SARSA(λ)"
                    manager.set_algo(algo)

                elif event.key == pygame.K_h:
                    overlay_heatmap = not overlay_heatmap

                elif event.key == pygame.K_p:
                    overlay_policy = not overlay_policy

                elif event.key == pygame.K_e:
                    manager.evaluate(episodes=30, max_steps=500)

                elif event.key == pygame.K_n:
                    _, last_runtime = manager.train_batch(episodes=episodes_per_batch)

                elif event.key == pygame.K_s:
                    ts = time.strftime("%Y%m%d-%H%M%S")
                    out = results_dir / f"screenshot_{ts}.png"
                    pygame.image.save(screen, out.as_posix())

                elif event.key == pygame.K_t:
                    base_env = env
                    new_grid, start, goal, key = create_destination_env(
                        base_env,
                        change_ratio=0.2,
                        move_key=True,
                        move_goal=True,
                    )
                    dest_env = DynamicMazeEnv(
                        grid=new_grid,
                        start=start,
                        goal=goal,
                        key=key,
                        reward_mode=config["reward_mode"],
                        seed=config["seed"] + 101,
                        step_success_prob=config["success_prob"],
                        penalty_reward=-10.0,
                        wall_penalty=-2.0,
                        move_cost=-1.0,
                        key_reward=20.0,
                        goal_reward=100.0,
                        energy_max=60,
                    )
                    on_destination = not on_destination
                    target = dest_env if on_destination else env
                    target.reset()
                    manager.set_env(target)
                    renderer.set_env(target)

                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    episodes_per_batch = min(200, episodes_per_batch + 5)

                elif event.key == pygame.K_MINUS:
                    episodes_per_batch = max(1, episodes_per_batch - 5)

                # manual movement
                elif event.key == pygame.K_UP:
                    current_env.step(0)
                elif event.key == pygame.K_DOWN:
                    current_env.step(1)
                elif event.key == pygame.K_LEFT:
                    current_env.step(2)
                elif event.key == pygame.K_RIGHT:
                    current_env.step(3)

        if training:
            _, last_runtime = manager.train_batch(episodes=episodes_per_batch)

        # automatic movement by learned policy
        now = time.time()
        if playing and manager.trained and now - last_move_time >= move_delay:
            action = manager.greedy_action()
            if action is not None:
                _, _, done, _ = current_env.step(action)
                last_move_time = now
                if done:
                    playing = False

        vgrid = manager.value_grid(has_key=0)
        pgrid = manager.policy_grid(has_key=0)

        renderer.set_overlays(
            value_grid=vgrid if overlay_heatmap else None,
            policy_grid=pgrid if overlay_policy else None,
        )

        info_lines = [
            f"Algo: {algo} | Reward: {config['reward_mode']}",
            f"Train: {'ON' if training else 'OFF'} | Play: {'ON' if playing else 'OFF'}",
            f"Batch: {episodes_per_batch} | Episodes: {manager.episodes_done}",
            f"Last dt: {last_runtime:.3f}s",
            f"HasKey: {current_env.has_key} | Energy: {current_env.energy}",
            f"Agent Pos: {current_env.agent_pos} | Goal: {current_env.goal}",
            f"On Destination: {on_destination}",
            "Keys: SPACE train | M autoplay | N train batch | R reset",
            "1/2/3 algo | H heatmap | P policy | V reward mode",
            "Arrows = manual move | T transfer | E evaluate | S screenshot",
        ]

        if manager.last_eval is not None:
            m = manager.last_eval
            info_lines += [
                f"Eval avgR {m['avg_reward']:.2f} ± {m['std_reward']:.2f}",
                f"Eval success {m['success_rate'] * 100:.1f}% | avg steps {m['avg_steps']:.1f}",
            ]

        renderer.draw(screen, info_lines)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    run_app()
