# Full file path: moneyverse/self_learning/self_learning_engine.py

import asyncio
from datetime import datetime, timedelta
from ai.rl_agent import RLTradingAgent
from rl_agent.manager_agent import ManagerAgent
from rl_agent.worker_agent import WorkerAgent
from rl_agent.meta_agent import MetaAgent
from utils.source_selector import SourceSelector
from utils.reward_calculator import calculate_reward
from utils.nav_monitor import NAVMonitor
from centralized_logger import CentralizedLogger
from database.async_db_handler import AsyncDBHandler
from market_data import MarketDataAPI

# Initialize centralized logging
logger = CentralizedLogger()

class SelfLearningEngine:
    """
    Self-Learning Engine for dynamic asset management with NAV optimization.
    Integrates real-time performance tracking, alerts, and adaptive trading strategy selection.
    """

    def __init__(self, db_handler: AsyncDBHandler, reinvestment_interval=3600):
        # Core AI and utility components
        self.db_handler = db_handler
        self.rl_agent = RLTradingAgent(environment="nav_optimization", model="PPO")
        self.manager_agent = ManagerAgent(nav_target_multiplier=2.0)
        self.worker_agents = [WorkerAgent() for _ in range(3)]
        self.meta_agent = MetaAgent(self.manager_agent, self.worker_agents)
        self.source_selector = SourceSelector(api_configs=db_handler.fetch("SELECT * FROM api_sources"))
        self.nav_monitor = NAVMonitor(self.rl_agent, db_handler, logger)
        self.market_data_api = MarketDataAPI()
        self.reinvestment_interval = reinvestment_interval
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(hours=1)
        self.profit_wallet = "non_swarm_wallet"
        self.configs = {}

    async def update_configuration(self):
        """Fetches updated configuration settings from the database for adaptive strategy adjustments."""
        configs = await self.db_handler.fetch("SELECT config_key, config_value FROM configurations")
        self.configs = {config["config_key"]: config["config_value"] for config in configs}

    async def optimize_nav(self):
        """Main loop for NAV optimization, leveraging RL-based decision-making and real-time adjustments."""
        while True:
            try:
                await self.update_configuration()  # Load updated configurations dynamically
                best_api = self.source_selector.choose_best_source()
                market_data = await self.source_selector.call_api(best_api)
                if not market_data:
                    logger.log("warning", "Market data fetch failed, using fallback.")
                    continue

                # ManagerAgent coordinates high-level NAV targets for WorkerAgents
                current_nav = self.calculate_nav()
                self.manager_agent.manage_portfolio(current_nav)

                # Each WorkerAgent executes trades to meet target NAV goals
                for worker in self.worker_agents:
                    worker.execute_trade(market_data)

                # Predict NAV trend using RL agent
                nav_prediction = self.rl_agent.predict_nav_trend(market_data)
                if nav_prediction < float(self.configs.get("rl_decision_threshold", 0.7)):
                    logger.log("warning", "RL decision threshold not met; skipping action.")
                    continue

                action = self.rl_agent.decide_action(market_data)
                if action:
                    reward = calculate_reward(action)
                    logger.log("info", f"Action taken with reward: {reward}")

            except Exception as e:
                logger.log("error", f"Error in optimize_nav loop: {str(e)}")

            await asyncio.sleep(10)  # Short delay before next cycle

    async def reinvestment_cycle(self):
        """Periodic reinvestment cycle based on profits and market condition updates."""
        while True:
            try:
                surplus = await self.nav_monitor.calculate_surplus()
                if surplus > 0:
                    self.wallet_manager.transfer_to_wallet(self.profit_wallet, surplus)
                    logger.log("info", f"Transferred {surplus} to profit wallet.")
                await asyncio.sleep(self.reinvestment_interval)

            except Exception as e:
                logger.log("error", f"Error in reinvestment cycle: {str(e)}")

    async def monitor_nav_and_alert(self):
        """Monitors NAV and sends alerts if NAV trends indicate significant growth or drop."""
        while datetime.now() < self.end_time:
            status = self.nav_monitor.countdown_status()
            nav_trend = self.nav_monitor.get_nav_trend()

            if nav_trend["predicted_loss"] >= 0.2:
                self.nav_monitor.send_notification("ALERT: Potential NAV drop exceeding 20% detected.")
            elif nav_trend["predicted_surplus"] > float(self.configs.get("goal_multiplier", 2.0)) * 0.2:
                self.nav_monitor.send_notification("SUCCESS: Projected NAV is on track to exceed goal by 20% surplus.")

            await asyncio.sleep(5)  # Check every 5 seconds

    def calculate_nav(self):
        """Placeholder method for NAV calculation. Returns a simulated NAV value."""
        return 10000  # Example NAV value

    async def run_engine(self):
        """Runs the full self-learning engine with all core functions in parallel."""
        await asyncio.gather(
            self.optimize_nav(),
            self.reinvestment_cycle(),
            self.monitor_nav_and_alert(),
            self.meta_agent.run_meta_learning()  # Meta-learning adjustment loop
        )

# Run the self-learning engine
if __name__ == "__main__":
    db_handler = AsyncDBHandler()
    engine = SelfLearningEngine(db_handler=db_handler)
    try:
        asyncio.run(engine.run_engine())
    except KeyboardInterrupt:
        logger.log("info", "Self-Learning Engine stopped.")
