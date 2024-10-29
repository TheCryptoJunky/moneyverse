# Full file path: moneyverse/self_learning/self_learning_engine.py

import asyncio
from ai.rl_agent import RLTradingAgent
from utils.source_selector import SourceSelector
from utils.reward_calculator import calculate_reward
from utils.nav_monitor import NAVMonitor
from centralized_logger import CentralizedLogger
from database.async_db_handler import AsyncDBHandler
from datetime import datetime, timedelta

# Centralized logging setup
logger = CentralizedLogger()

class SelfLearningEngine:
    """
    AI-driven Self-Learning Engine for dynamic asset management with a focus on NAV optimization.
    Integrates real-time performance tracking, alerts, and adaptive trading strategy selection.
    """

    def __init__(self, db_handler: AsyncDBHandler, reinvestment_interval=3600):
        # Initialize core AI and utility components
        self.db_handler = db_handler
        self.rl_agent = RLTradingAgent(environment="nav_optimization", model="PPO")
        self.source_selector = SourceSelector(api_configs=db_handler.fetch("SELECT * FROM api_sources"))
        self.nav_monitor = NAVMonitor(self.rl_agent, db_handler, logger)
        self.reinvestment_interval = reinvestment_interval
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(hours=1)
        self.profit_wallet = "non_swarm_wallet"

    async def update_configuration(self):
        """
        Fetches updated configuration settings from the database for adaptive strategy.
        """
        configs = await self.db_handler.fetch("SELECT config_key, config_value FROM configurations")
        self.configs = {config["config_key"]: config["config_value"] for config in configs}

    async def optimize_nav(self):
        """
        Main loop for NAV optimization, leveraging RL-based decision-making and real-time adjustments.
        """
        while True:
            try:
                await self.update_configuration()  # Reload configuration dynamically

                # Step 1: Fetch and assess market data
                best_api = self.source_selector.choose_best_source()
                market_data = await self.source_selector.call_api(best_api)
                if market_data is None:
                    logger.log("warning", "Market data fetch failed, using fallback.")
                    continue

                # Step 2: Predict NAV trend and decide next action
                nav_prediction = self.rl_agent.predict_nav_trend(market_data)
                if nav_prediction < float(self.configs["rl_decision_threshold"]):
                    logger.log("warning", "RL decision threshold not met; skipping action.")
                    continue

                # Step 3: Execute the AI-driven trading action
                action = self.rl_agent.decide_action(market_data)
                if action:
                    reward = calculate_reward(action)
                    logger.log("info", f"Action taken with reward: {reward}")

            except Exception as e:
                logger.log("error", f"Error in optimize_nav loop: {str(e)}")

            await asyncio.sleep(10)  # Short delay before next cycle

    async def reinvestment_cycle(self):
        """
        Periodic reinvestment cycle based on profits and market condition updates.
        """
        while True:
            try:
                # Calculate surplus and reinvest to reach target NAV
                surplus = await self.nav_monitor.calculate_surplus()
                if surplus > 0:
                    self.wallet_manager.transfer_to_wallet(self.profit_wallet, surplus)
                    logger.log("info", f"Transferred {surplus} to profit wallet.")

                await asyncio.sleep(self.reinvestment_interval)

            except Exception as e:
                logger.log("error", f"Error in reinvestment cycle: {str(e)}")

    async def monitor_nav_and_alert(self):
        """
        Monitors NAV and sends alerts if NAV trends indicate significant growth or drop.
        """
        while datetime.now() < self.end_time:
            status = self.nav_monitor.countdown_status()
            nav_trend = self.nav_monitor.get_nav_trend()

            # Send alerts based on trends
            if nav_trend["predicted_loss"] >= 0.2:
                self.nav_monitor.send_notification("ALERT: Potential NAV drop exceeding 20% detected.")
            elif nav_trend["predicted_surplus"] > float(self.configs["goal_multiplier"]) * 0.2:
                self.nav_monitor.send_notification("SUCCESS: Projected NAV is on track to exceed goal by 20% surplus.")

            await asyncio.sleep(5)  # Check every 5 seconds

    async def run_engine(self):
        """
        Runs the full self-learning engine with all core functions in parallel.
        """
        await asyncio.gather(
            self.optimize_nav(),
            self.reinvestment_cycle(),
            self.monitor_nav_and_alert()
        )

# Run the self-learning engine
if __name__ == "__main__":
    db_handler = AsyncDBHandler()
    engine = SelfLearningEngine(db_handler=db_handler)
    try:
        asyncio.run(engine.run_engine())
    except KeyboardInterrupt:
        logger.log("info", "Self-Learning Engine stopped.")
