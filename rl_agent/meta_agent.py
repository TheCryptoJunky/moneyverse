# Full file path: /rl_agent/meta_agent.py

from datetime import datetime, timedelta
from centralized_logger import CentralizedLogger
from rl_agent.manager_agent import ManagerAgent
from rl_agent.worker_agent import WorkerAgent
import random
import numpy as np
import asyncio

logger = CentralizedLogger()

class MetaAgent:
    """
    Meta-Learning Agent for dynamic hyperparameter tuning, multi-objective optimization, and performance tracking.
    Adjusts parameters based on real-time performance metrics for adaptive learning.
    """

    def __init__(self, manager_agent: ManagerAgent, worker_agents: list, adjustment_interval=3600):
        """
        Initialize MetaAgent with a ManagerAgent, WorkerAgents, and meta-learning settings.

        Args:
            manager_agent (ManagerAgent): The primary agent managing high-level goals.
            worker_agents (list): List of WorkerAgents executing trades.
            adjustment_interval (int): Time in seconds between hyperparameter adjustments.
        """
        self.manager_agent = manager_agent
        self.worker_agents = worker_agents
        self.adjustment_interval = adjustment_interval
        self.last_adjustment_time = datetime.now()

        # Initial Hyperparameters
        self.hyperparams = {
            "manager_learning_rate": 0.001,
            "worker_learning_rate": 0.001,
            "exploration_rate": 0.1,
            "discount_factor": 0.95,
            "nav_growth_weight": 0.7,
            "resource_efficiency_weight": 0.3
        }

        # Performance metrics history
        self.performance_history = {
            "nav_growth_rate": [],
            "trade_success_rate": [],
            "average_reward": []
        }

    def track_performance(self, nav_growth, trade_success_rate, avg_reward):
        """
        Records performance metrics to maintain a history for trend analysis.

        Args:
            nav_growth (float): NAV growth rate.
            trade_success_rate (float): Success rate of trades.
            avg_reward (float): Average reward of recent trades.
        """
        self.performance_history["nav_growth_rate"].append(nav_growth)
        self.performance_history["trade_success_rate"].append(trade_success_rate)
        self.performance_history["average_reward"].append(avg_reward)

        # Limit history length to a manageable size
        max_history_length = 100
        for key in self.performance_history:
            if len(self.performance_history[key]) > max_history_length:
                self.performance_history[key] = self.performance_history[key][-max_history_length:]

    def analyze_performance_trends(self):
        """
        Analyzes recent trends in NAV growth, trade success rate, and reward to adjust hyperparameters.
        """
        avg_nav_growth = np.mean(self.performance_history["nav_growth_rate"][-10:])
        avg_trade_success_rate = np.mean(self.performance_history["trade_success_rate"][-10:])
        avg_reward = np.mean(self.performance_history["average_reward"][-10:])

        # Adjust exploration rate based on trade success trends
        if avg_trade_success_rate < 0.6:
            self.hyperparams["exploration_rate"] = min(self.hyperparams["exploration_rate"] + 0.05, 0.3)
            logger.log("info", f"Increased exploration rate to {self.hyperparams['exploration_rate']} due to low success rate.")
        elif avg_trade_success_rate > 0.8:
            self.hyperparams["exploration_rate"] = max(self.hyperparams["exploration_rate"] - 0.05, 0.05)
            logger.log("info", f"Decreased exploration rate to {self.hyperparams['exploration_rate']} due to high success rate.")

        # Adjust learning rate for ManagerAgent based on NAV growth trends
        if avg_nav_growth < 1.0:
            self.hyperparams["manager_learning_rate"] = min(self.hyperparams["manager_learning_rate"] + 0.0001, 0.01)
            logger.log("info", f"Increased manager learning rate to {self.hyperparams['manager_learning_rate']} due to low NAV growth.")
        elif avg_nav_growth > 1.5:
            self.hyperparams["manager_learning_rate"] = max(self.hyperparams["manager_learning_rate"] - 0.0001, 0.0005)
            logger.log("info", f"Decreased manager learning rate to {self.hyperparams['manager_learning_rate']} due to high NAV growth.")

    def apply_adjustments(self):
        """
        Updates the ManagerAgent and WorkerAgents with adjusted hyperparameters.
        """
        self.manager_agent.learning_rate = self.hyperparams["manager_learning_rate"]
        for worker in self.worker_agents:
            worker.learning_rate = self.hyperparams["worker_learning_rate"]
            worker.exploration_rate = self.hyperparams["exploration_rate"]
            worker.discount_factor = self.hyperparams["discount_factor"]

        logger.log("info", "Applied updated hyperparameters to ManagerAgent and WorkerAgents.")

    async def run_meta_learning(self):
        """
        Main loop to periodically evaluate performance and apply meta-learning adjustments.
        """
        while True:
            time_elapsed = (datetime.now() - self.last_adjustment_time).total_seconds()
            if time_elapsed >= self.adjustment_interval:
                # Fetch performance metrics (replace with actual metrics gathering)
                nav_growth = self.manager_agent.calculate_nav_growth()
                trade_success_rate = self.calculate_trade_success_rate()
                avg_reward = self.calculate_average_reward()

                # Track metrics, analyze trends, and apply adjustments
                self.track_performance(nav_growth, trade_success_rate, avg_reward)
                self.analyze_performance_trends()
                self.apply_adjustments()
                
                self.last_adjustment_time = datetime.now()

            await asyncio.sleep(60)  # Check every minute

    def calculate_trade_success_rate(self):
        """
        Calculates success rate of WorkerAgent trades to inform adjustment logic.
        
        Returns:
            float: Success rate across WorkerAgents.
        """
        success_count = sum(1 for worker in self.worker_agents if worker.last_trade_profit > 0)
        return success_count / len(self.worker_agents) if self.worker_agents else 0.0

    def calculate_average_reward(self):
        """
        Computes the average reward for recent trades across WorkerAgents.

        Returns:
            float: Average trade reward across WorkerAgents.
        """
        rewards = [worker.last_trade_profit for worker in self.worker_agents]
        return sum(rewards) / len(rewards) if rewards else 0.0
