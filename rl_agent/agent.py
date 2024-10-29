# Full file path: moneyverse/rl_agent/agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .dqn import DQN
from .pgm import PGM
from .mev_strategies import MEVStrategies
from .wallet_swarm import WalletSwarm
from utils.nav_monitor import NAVMonitor
from utils.performance_tracker import PerformanceTracker

class Agent(nn.Module):
    def __init__(self, state_dim, action_dim, nav_calculator, performance_tracker, nav_monitor, use_dynamic_selection=True):
        super(Agent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.nav_calculator = nav_calculator
        self.performance_tracker = performance_tracker
        self.nav_monitor = nav_monitor
        self.dqn = DQN(state_dim, action_dim)
        self.pgm = PGM(state_dim, action_dim)
        self.mev_strategies = MEVStrategies()
        self.wallet_swarm = WalletSwarm()
        self.writer = SummaryWriter(log_dir="runs/agent_logs")
        self.use_dynamic_selection = use_dynamic_selection  # Toggle between DQN and PGM based on recent performance

    def select_action(self, state):
        """
        Select an action using either DQN or PGM based on dynamic selection or predefined logic.
        """
        if self.use_dynamic_selection:
            recent_performance = self.performance_tracker.get_recent_performance()
            action = self.dqn.select_action(state) if recent_performance > 0.7 else self.pgm.select_action(state)
        else:
            action = self.dqn.select_action(state)
        
        self.writer.add_scalar("Agent/Selected_Action", action, global_step=self.performance_tracker.current_step())
        return action

    def update(self, state, action, reward, next_state):
        """
        Update DQN and PGM models with the given transition (state, action, reward, next_state).
        """
        self.dqn.update(state, action, reward, next_state)
        self.pgm.update(state, action, reward, next_state)
        self.writer.add_scalar("Agent/Reward", reward, global_step=self.performance_tracker.current_step())

    def get_mev_strategy(self, state):
        """
        Select an MEV strategy based on the state and recent NAV trends.
        """
        nav_trend = self.nav_monitor.get_trend()  # Get real-time NAV trend
        mev_strategy = self.mev_strategies.get_mev_strategy(state, nav_trend)
        self.writer.add_text("Agent/Selected_MEV_Strategy", mev_strategy, global_step=self.performance_tracker.current_step())
        return mev_strategy

    def execute_mev_strategy(self, mev_strategy):
        """
        Execute the chosen MEV strategy using the WalletSwarm.
        """
        result = self.wallet_swarm.execute_mev_strategy(mev_strategy)
        self.writer.add_scalar("Agent/MEV_Execution_Success", result, global_step=self.performance_tracker.current_step())
        return result

    def calculate_nav(self):
        """
        Calculate the current NAV using the NAVCalculator.
        """
        nav = self.nav_calculator.calculate_nav()
        self.writer.add_scalar("Agent/NAV", nav, global_step=self.performance_tracker.current_step())
        return nav

    def track_performance(self):
        """
        Track and log performance using the PerformanceTracker.
        """
        self.performance_tracker.track_performance()
        self.writer.add_scalar("Agent/Performance", self.performance_tracker.current_performance(), global_step=self.performance_tracker.current_step())
