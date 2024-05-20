# Mario AI Project

## Overview

This project aims to train an AI agent to play Super Mario Bros using reinforcement learning. Utilizing the Stable Baselines3 library with a PPO model, the agent learns to navigate levels by maximizing its score while minimizing deaths.

## Getting Started

### Prerequisites

- Python 3.8
- PyTorch
- Stable Baselines3
- Gym-Super-Mario-Bros
- NES-Py (dependency of Gym-Super-Mario-Bros)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Indrr27/MarioAI.git
cd MarioAI
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

### Files Description

- `environment_setup.py`: Contains functions to configure the Mario environment for training and evaluation.
- `callbacks.py`: Defines a custom callback for periodic saving during training.
- `main.py`: The main script for training and testing the Mario AI agent.

## Training the AI

To start training the AI agent with default settings, run:

```bash
python main.py
```

This will initiate training based on the configurations set within `main.py`, including the environment setup and model parameters. The AI will learn over 1,000,000 steps and save its progress at intervals of 10,000 steps.

## Evaluating the Model

After training, the model's performance can be evaluated by running the latter part of `main.py`, where the trained model is loaded and used to play the game. This section visualizes the agent's gameplay in real-time.
# Mario AI Machine Learning Progress

## Mario at 10,000 Steps
[![Mario at 10,000 Steps](https://img.youtube.com/vi/NJACjJI1P9A/0.jpg)](https://www.youtube.com/watch?v=NJACjJI1P9A)
Watch Mario's progress at 10,000 steps.

## Mario at 500,000 Steps
[![Mario at 500,000 Steps](https://img.youtube.com/vi/JynbLB7jYjE/0.jpg)](https://www.youtube.com/watch?v=JynbLB7jYjE)
Watch Mario's progress at 500,000 steps.

## Mario at 1,000,000 Steps
[![Mario at 1,000,000 Steps](https://img.youtube.com/vi/B4arRy4jwDs/0.jpg)](https://www.youtube.com/watch?v=B4arRy4jwDs)
Watch Mario's progress at 1,000,000 steps.

