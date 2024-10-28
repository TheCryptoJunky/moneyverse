import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .dqn import DQN
from .pgm import PGM
from .mev_strategies import MEVStrategies
from .wallet_swarm import WalletSwarm

class Agent(nn.Module):
    def __init__(self, state_dim, action_dim, nav_calculator, performance_tracker):
        super(Agent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.nav_calculator = nav_calculator
        self.performance_tracker = performance_tracker
        self.dqn = DQN(state_dim, action_dim)
        self.pgm = PGM(state_dim, action_dim)
        self.mev_strategies = MEVStrategies()
        self.wallet_swarm = WalletSwarm()

    def select_action(self, state):
        # Select action using DQN or PGM
        action = self.dqn.select_action(state)
        return action

    def update(self, state, action, reward, next_state):
        # Update DQN and PGM
        self.dqn.update(state, action, reward, next_state)
        self.pgm.update(state, action, reward, next_state)

    def get_mev_strategy(self, state):
        # Get MEV strategy using MEVStrategies
        mev_strategy = self.mev_strategies.get_mev_strategy(state)
        return mev_strategy

    def execute_mev_strategy(self, mev_strategy):
        # Execute MEV strategy using WalletSwarm
        self.wallet_swarm.execute_mev_strategy(mev_strategy)

    def calculate_nav(self):
        # Calculate NAV using NAVCalculator
        nav = self.nav_calculator.calculate_nav()
        return nav

    def track_performance(self):
        # Track performance using PerformanceTracker
        self.performance_tracker.track_performance()
