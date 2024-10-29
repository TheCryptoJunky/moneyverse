# Full file path: /rl_agent/manager_agent.py

from datetime import datetime, timedelta
from centralized_logger import CentralizedLogger
from rl_agent.worker_agent import WorkerAgent

logger = CentralizedLogger()

class ManagerAgent:
    """
    Manages high-level trading strategies, NAV goals, and directs WorkerAgents.
    Uses adaptive strategies for cumulative NAV growth and real-time adjustments
    based on performance metrics.
    """

    def __init__(self, nav_target_multiplier=2.0, update_interval=3600, initial_num_agents=3, volatility_threshold=0.05):
        """
        Initialize the ManagerAgent with NAV target multiplier and update interval.
        
        Args:
            nav_target_multiplier (float): Target multiplier for NAV growth per hour.
            update_interval (int): Interval in seconds for updating WorkerAgent directives.
            initial_num_agents (int): Initial number of WorkerAgents under ManagerAgent's control.
            volatility_threshold (float): Threshold to adjust NAV targets based on market volatility.
        """
        self.nav_target_multiplier = nav_target_multiplier
        self.update_interval = update_interval
        self.volatility_threshold = volatility_threshold
        self.worker_agents = []
        self.initial_nav = None
        self.last_update_time = datetime.now()
        self.create_worker_agents(initial_num_agents)

    def create_worker_agents(self, num_agents):
        """Initialize specified number of WorkerAgents under Manager control."""
        for _ in range(num_agents):
            worker = WorkerAgent()
            self.worker_agents.append(worker)
        logger.log("info", f"Initialized {num_agents} WorkerAgents under ManagerAgent")

    def scale_worker_agents(self, performance_score):
        """
        Dynamically scales the number of WorkerAgents based on recent performance.
        
        Args:
            performance_score (float): Indicator of WorkerAgent success; scales agents accordingly.
        """
        target_num_agents = max(1, int(len(self.worker_agents) * (1 + performance_score)))
        if target_num_agents != len(self.worker_agents):
            self.worker_agents = self.worker_agents[:target_num_agents] + \
                [WorkerAgent() for _ in range(target_num_agents - len(self.worker_agents))]
            logger.log("info", f"Adjusted WorkerAgents to {len(self.worker_agents)} based on performance score.")

    def adjust_nav_target(self, market_volatility):
        """
        Adjusts NAV target based on recent market volatility.
        
        Args:
            market_volatility (float): Measure of recent market volatility.
        """
        if market_volatility > self.volatility_threshold:
            self.nav_target_multiplier *= 0.95  # Slightly reduce target in high volatility
            logger.log("warning", f"High market volatility detected. Adjusted NAV target multiplier to {self.nav_target_multiplier}")
        else:
            self.nav_target_multiplier *= 1.05  # Slightly increase target in low volatility
            logger.log("info", f"Stable market detected. Adjusted NAV target multiplier to {self.nav_target_multiplier}")

    def update_directives(self, current_nav, market_volatility):
        """
        Updates directives for WorkerAgents based on NAV and market volatility.
        
        Args:
            current_nav (float): Current NAV of the wallet swarm.
            market_volatility (float): Current market volatility.
        """
        # Adjust NAV target based on market conditions
        self.adjust_nav_target(market_volatility)
        
        # Set the target NAV for WorkerAgents
        target_nav = current_nav * self.nav_target_multiplier
        logger.log("info", f"Setting target NAV: {target_nav} for WorkerAgents")

        # Assign updated NAV targets to WorkerAgents
        for worker in self.worker_agents:
            worker.set_target_nav(target_nav)

    def manage_portfolio(self, current_nav, market_volatility, performance_score):
        """
        Manages portfolio, updating WorkerAgent tasks as needed and adjusting NAV targets.
        
        Args:
            current_nav (float): Current NAV of the wallet swarm.
            market_volatility (float): Current measure of market volatility.
            performance_score (float): Performance score for scaling WorkerAgents.
        """
        time_elapsed = (datetime.now() - self.last_update_time).total_seconds()
        if time_elapsed >= self.update_interval:
            self.update_directives(current_nav, market_volatility)
            self.scale_worker_agents(performance_score)
            self.last_update_time = datetime.now()
            logger.log("info", "ManagerAgent updated WorkerAgents with new directives")

    def calculate_reward(self, current_nav, elapsed_time_hours):
        """
        Reward function focused on long-term NAV growth with adaptive target scaling.
        
        Args:
            current_nav (float): The current NAV of the wallet swarm.
            elapsed_time_hours (float): Time in hours since the start of tracking.

        Returns:
            float: Calculated reward based on NAV growth compared to adaptive target.
        """
        if self.initial_nav is None:
            self.initial_nav = current_nav
            return 0.0

        # Calculate growth factor and adaptive target growth
        growth_factor = current_nav / self.initial_nav
        target_growth = self.nav_target_multiplier ** (elapsed_time_hours / 24)

        # Reward is based on exceeding adaptive growth target, scaled by performance
        reward = max(0, (growth_factor - target_growth) * 100)
        logger.log("info", f"ManagerAgent calculated reward: {reward}")
        return reward

    def reset_tracking(self):
        """Resets the tracking for a new cumulative NAV growth period."""
        self.initial_nav = None
        logger.log("info", "ManagerAgent reset NAV growth tracking for a new period")
