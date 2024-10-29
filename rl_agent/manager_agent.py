# Full file path: /rl_agent/manager_agent.py

from datetime import datetime, timedelta
from centralized_logger import CentralizedLogger
from rl_agent.worker_agent import WorkerAgent

logger = CentralizedLogger()

class ManagerAgent:
    """
    Manages high-level trading strategies, NAV targets, and directs WorkerAgents.
    Incorporates adaptive strategies for cumulative NAV growth, real-time market conditions,
    and agent scaling based on performance metrics.
    """

    def __init__(self, nav_target_multiplier=2.0, update_interval=3600, initial_num_agents=3, volatility_threshold=0.05):
        """
        Initializes ManagerAgent with NAV target multiplier, update interval, initial WorkerAgents,
        and market volatility threshold.

        Args:
            nav_target_multiplier (float): Target NAV growth multiplier per hour.
            update_interval (int): Interval in seconds for updating WorkerAgent directives.
            initial_num_agents (int): Initial number of WorkerAgents under ManagerAgent's control.
            volatility_threshold (float): Threshold for adjusting NAV targets based on market volatility.
        """
        self.nav_target_multiplier = nav_target_multiplier
        self.update_interval = update_interval
        self.volatility_threshold = volatility_threshold
        self.worker_agents = []
        self.initial_nav = None
        self.last_update_time = datetime.now()
        self.create_worker_agents(initial_num_agents)

    def create_worker_agents(self, num_agents):
        """Initializes the specified number of WorkerAgents under Manager control."""
        self.worker_agents = [WorkerAgent() for _ in range(num_agents)]
        logger.log("info", f"Initialized {num_agents} WorkerAgents under ManagerAgent.")

    def scale_worker_agents(self, performance_score):
        """
        Dynamically adjusts the number of WorkerAgents based on performance metrics.

        Args:
            performance_score (float): Indicator of WorkerAgent success; scales agents accordingly.
        """
        target_num_agents = max(1, int(len(self.worker_agents) * (1 + performance_score)))
        current_num_agents = len(self.worker_agents)
        
        if target_num_agents != current_num_agents:
            if target_num_agents > current_num_agents:
                # Add new agents if needed
                self.worker_agents.extend(WorkerAgent() for _ in range(target_num_agents - current_num_agents))
            else:
                # Remove excess agents if over-performing
                self.worker_agents = self.worker_agents[:target_num_agents]
                
            logger.log("info", f"Scaled WorkerAgents to {target_num_agents} based on performance score.")

    def adjust_nav_target(self, market_volatility):
        """
        Adjusts NAV target multiplier based on recent market volatility.

        Args:
            market_volatility (float): Measure of recent market volatility.
        """
        if market_volatility > self.volatility_threshold:
            # Reduce NAV target in volatile conditions
            self.nav_target_multiplier *= 0.95
            logger.log("warning", f"High market volatility detected. Adjusted NAV target multiplier to {self.nav_target_multiplier}.")
        else:
            # Increase NAV target in stable conditions
            self.nav_target_multiplier *= 1.05
            logger.log("info", f"Stable market detected. Adjusted NAV target multiplier to {self.nav_target_multiplier}.")

    def update_directives(self, current_nav, market_volatility):
        """
        Sets target NAV for WorkerAgents based on current NAV and market conditions.

        Args:
            current_nav (float): Current NAV of the wallet swarm.
            market_volatility (float): Current market volatility.
        """
        # Adjust target multiplier based on market conditions
        self.adjust_nav_target(market_volatility)

        # Set the target NAV for WorkerAgents
        target_nav = current_nav * self.nav_target_multiplier
        logger.log("info", f"Setting target NAV: {target_nav} for WorkerAgents.")

        # Update each WorkerAgent with the new target NAV
        for worker in self.worker_agents:
            worker.set_target_nav(target_nav)

    def manage_portfolio(self, current_nav, market_volatility, performance_score):
        """
        Manages the portfolio by updating WorkerAgent tasks as needed, adjusting NAV targets, and scaling agents.

        Args:
            current_nav (float): Current NAV of the wallet swarm.
            market_volatility (float): Current market volatility.
            performance_score (float): Performance score for scaling WorkerAgents.
        """
        time_elapsed = (datetime.now() - self.last_update_time).total_seconds()
        
        if time_elapsed >= self.update_interval:
            self.update_directives(current_nav, market_volatility)
            self.scale_worker_agents(performance_score)
            self.last_update_time = datetime.now()
            logger.log("info", "ManagerAgent updated WorkerAgents with new directives.")

    def calculate_reward(self, current_nav, elapsed_time_hours):
        """
        Reward function focusing on long-term NAV growth with adaptive target scaling.

        Args:
            current_nav (float): The current NAV of the wallet swarm.
            elapsed_time_hours (float): Time in hours since the start of tracking.

        Returns:
            float: Calculated reward based on NAV growth relative to adaptive target.
        """
        if self.initial_nav is None:
            # Initialize tracking with the starting NAV
            self.initial_nav = current_nav
            return 0.0

        # Calculate growth factor and the adaptive target growth
        growth_factor = current_nav / self.initial_nav
        adaptive_target_growth = self.nav_target_multiplier ** (elapsed_time_hours / 24)

        # Reward is the difference between actual growth and the adaptive target growth
        reward = max(0, (growth_factor - adaptive_target_growth) * 100)
        logger.log("info", f"ManagerAgent calculated reward: {reward}.")
        return reward

    def reset_tracking(self):
        """Resets NAV growth tracking for a new cumulative growth period."""
        self.initial_nav = None
        logger.log("info", "ManagerAgent reset NAV growth tracking for a new period.")
