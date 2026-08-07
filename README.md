# Dynamic Maze Reinforcement Learning Project

This project is a reinforcement learning final project that compares:

- **Value Iteration**
- **Q-Learning**
- **SARSA(λ)**

It includes:

- dynamic maze generation
- stochastic movement
- shaped and sparse rewards
- transfer learning experiments
- a GUI for training and visualization
- automated experiment scripts
- unit tests

---

## Project Structure
RL_FinalProject_40413824/

├── agents/

├── environments/

├── experiments/

├── gui/

├── transfer/

├── tests/

├── main.py

├── README.md

├── requirements.txt

└── report.pdf

---

## Features

### Environment
- random maze generation
- deterministic seeding
- start, goal, and key placement
- penalty cells
- stochastic transitions

### Algorithms
- **Value Iteration**
- **Q-Learning**
- **SARSA(λ)**

### Reward Modes
- shaped reward
- sparse reward

### Transfer Learning
- train on a source maze
- adapt to a destination maze
- compare transfer vs. training from scratch

### GUI
- select algorithm
- train agent
- autoplay learned policy
- manual movement
- visualize policy and values

### Experiments
- batch runs
- config-based execution
- saved metrics and plots

---

## Installation

Install dependencies:
pip install -r requirements.txt

---

## How to Run

### 1) Run the GUI
python -m gui.app

#### GUI Controls
- `1` → Value Iteration
- `2` → Q-Learning
- `3` → SARSA(λ)
- `SPACE` → train
- `M` → autoplay learned policy
- Arrow keys → manual movement
- `N` → next batch / step
- `T` → transfer mode
- `R` → reset environment
- `V` → toggle reward mode

---

### 2) Run Experiments

bash
python -m experiments.run_experiments --student-id 40413824

This should run the project with the derived settings from the student ID.

Outputs are usually saved in a results folder such as:

text
results/exp_<timestamp>/

---

### 3) Run Tests

bash
pytest -q

---

## Algorithms

### Value Iteration
A model-based planning algorithm that computes the optimal value function and greedy policy.

### Q-Learning
An off-policy learning algorithm that learns action-values from experience.

### SARSA(λ)
An on-policy learning algorithm with eligibility traces.

---

## Transfer Learning

The transfer-learning part trains on one maze and reuses learned knowledge in a modified destination maze.

It helps compare:

- training from scratch
- transfer initialization
- adaptation speed

---

## Output Files

Typical outputs may include:

- `config.json`
- `metrics_*.json`
- `trainstats_*.csv`
- `qtable_*.npy`
- maze images
- value heatmaps
- policy visualizations

---

## Notes

- Value Iteration can produce a policy without training episodes.
- Q-Learning and SARSA(λ) require training before good policies appear.
- The GUI supports both manual play and policy autoplay.
- The project is intended to be reproducible using the student-ID-based seed rule.

---

## Example Commands

Run GUI:

bash
python -m gui.app

Run experiments:

bash
python -m experiments.run_experiments --student-id 40413824

Run tests:

bash
pytest -q
