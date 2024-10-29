# Full file path: /bot/src/rl_agent/utils.py

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Tuple, Union

# Configure logging for utils
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def choose_action_epsilon_greedy(model: nn.Module, state: np.ndarray, epsilon: float, device: torch.device) -> int:
    """
    Select an action using the epsilon-greedy strategy to balance exploration and exploitation.
    
    Args:
        model (nn.Module): The Q-network model.
        state (np.ndarray): The current environment state.
        epsilon (float): Exploration rate; probability of choosing a random action.
        device (torch.device): Device to use for computations.

    Returns:
        int: The chosen action.
    """
    try:
        if np.random.rand() < epsilon:
            action = np.random.randint(0, model.action_dim)
            logger.debug(f"Random action chosen: {action}")
            return action
        else:
            state = torch.tensor(state, dtype=torch.float32).to(device)
            q_values = model(state)
            action = torch.argmax(q_values).item()
            logger.debug(f"Greedy action chosen: {action}")
            return action
    except Exception as e:
        logger.error(f"Error in epsilon-greedy action selection: {e}")
        raise

def choose_action_softmax(model: nn.Module, state: np.ndarray, epsilon: float, device: torch.device) -> int:
    """
    Select an action using a softmax policy, which introduces randomness weighted by action values.

    Args:
        model (nn.Module): The Q-network model.
        state (np.ndarray): The current environment state.
        epsilon (float): Exploration rate; higher values increase randomness.
        device (torch.device): Device to use for computations.

    Returns:
        int: The selected action.
    """
    try:
        state = torch.tensor(state, dtype=torch.float32).to(device)
        logits = model(state)
        probs = torch.softmax(logits, dim=0)
        if np.random.rand() < epsilon:
            action = torch.multinomial(probs, 1).item()
            logger.debug(f"Softmax random action chosen: {action}")
            return action
        else:
            action = torch.argmax(probs).item()
            logger.debug(f"Softmax greedy action chosen: {action}")
            return action
    except Exception as e:
        logger.error(f"Error in softmax action selection: {e}")
        raise

def calculate_epsilon(epsilon_start: float, epsilon_end: float, epsilon_decay: float, steps: int) -> float:
    """
    Calculate a decayed epsilon value for exploration-exploitation balance over training steps.

    Args:
        epsilon_start (float): Initial epsilon (high exploration).
        epsilon_end (float): Final epsilon (low exploration).
        epsilon_decay (float): Decay rate per step.
        steps (int): Current step in training.

    Returns:
        float: The decayed epsilon.
    """
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-steps * epsilon_decay)
    logger.debug(f"Epsilon calculated: {epsilon}")
    return epsilon

def calculate_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Calculate the entropy of a probability distribution for exploration encouragement.

    Args:
        probs (torch.Tensor): Probability distribution of actions.

    Returns:
        torch.Tensor: Average entropy across the distribution, encouraging diversity in actions.
    """
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    logger.debug(f"Entropy calculated: {entropy.mean().item()}")
    return entropy.mean()

def normalize_state(states: Union[torch.Tensor, np.ndarray], device: torch.device) -> torch.Tensor:
    """
    Normalize a batch of states to zero mean and unit variance for consistent input scaling.

    Args:
        states (torch.Tensor | np.ndarray): Batch of environment states.
        device (torch.device): Device for computation.

    Returns:
        torch.Tensor: Normalized batch of states.
    """
    try:
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32)
        states = states.to(device)
        normalized_states = (states - states.mean(dim=0)) / (states.std(dim=0) + 1e-8)
        logger.debug(f"Normalized states calculated.")
        return normalized_states
    except Exception as e:
        logger.error(f"Error in state normalization: {e}")
        raise

# Additional utility conversions

def to_tensor(x: np.ndarray) -> torch.Tensor:
    """Convert numpy array to tensor."""
    return torch.tensor(x, dtype=torch.float32)

def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array."""
    return x.cpu().numpy()
