# /bot/src/rl_agent/utils.py

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

def choose_action_epsilon_greedy(model: nn.Module, state: np.ndarray, epsilon: float, device: torch.device) -> int:
    """
    Choose an action using epsilon-greedy strategy.

    Args:
    - model (nn.Module): The Q-network model.
    - state (np.ndarray): The current state.
    - epsilon (float): The probability of choosing a random action.
    - device (torch.device): The device to use for computations.

    Returns:
    - int: The chosen action.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(0, model.action_dim)
    else:
        state = torch.tensor(state, dtype=torch.float32).to(device)
        q_values = model(state)
        return torch.argmax(q_values).item()

def choose_action_softmax(model: nn.Module, state: np.ndarray, epsilon: float, device: torch.device) -> int:
    """
    Choose an action using softmax strategy.

    Args:
    - model (nn.Module): The Q-network model.
    - state (np.ndarray): The current state.
    - epsilon (float): The probability of choosing a random action.
    - device (torch.device): The device to use for computations.

    Returns:
    - int: The chosen action.
    """
    state = torch.tensor(state, dtype=torch.float32).to(device)
    logits = model(state)
    probs = torch.softmax(logits, dim=0)
    if np.random.rand() < epsilon:
        return torch.multinomial(probs, 1).item()
    else:
        return torch.argmax(probs).item()

def calculate_epsilon(epsilon_start: float, epsilon_end: float, epsilon_decay: float, steps: int) -> float:
    """
    Calculate the current epsilon value using exponential decay.

    Args:
    - epsilon_start (float): The initial epsilon value.
    - epsilon_end (float): The final epsilon value.
    - epsilon_decay (float): The decay rate.
    - steps (int): The current step.

    Returns:
    - float: The current epsilon value.
    """
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-steps * epsilon_decay)
    return epsilon

def calculate_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Calculate the entropy of a probability distribution.

    Args:
    - probs (torch.Tensor): The probability distribution.

    Returns:
    - torch.Tensor: The entropy of the distribution.
    """
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return entropy.mean()

def normalize_state(states: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Normalize a batch of states by subtracting the mean and dividing by the standard deviation.

    Args:
    - states (torch.Tensor): The batch of states.
    - device (torch.device): The device to use for computations.

    Returns:
    - torch.Tensor: The normalized states.
    """
    states = states.to(device)
    states = (states - states.mean(dim=0)) / (states.std(dim=0) + 1e-8)
    return states

# Utilities

import numpy as np

def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

def to_numpy(x):
    return x.numpy()

# Changes:
# 1. Implemented utility functions for tensor and numpy conversions.
