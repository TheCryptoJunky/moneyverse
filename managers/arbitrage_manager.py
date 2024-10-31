# Full file path: /moneyverse/managers/arbitrage_manager.py

import asyncio
from centralized_logger import CentralizedLogger
from ai.agents.rl_agent import RLTradingAgent
from src.safety.safety_manager import SafetyManager
from src.list_manager import ListManager
from src.utils.error_handler import handle_errors
from src.trading.trade_executor import TradeExecutor

logger = CentralizedLogger()
rl_agent = RLTradingAgent(environment="arbitrage_trading")
safety_manager = SafetyManager()
trade_executor = TradeExecutor()
list_manager = ListManager()

class ArbitrageManager:
    def __init__(self):
        self.bots = []
        self.active_tasks = []

    # ---------------- Opportunity Handler Starts Here ----------------
    def handle_arbitrage_opportunity(self, opportunity: dict):
        """
        Responds to detected arbitrage opportunities.
        
        Args:
        - opportunity (dict): Opportunity data detected by the MempoolMonitor.
        """
        asset = opportunity.get("asset")
        spread = opportunity.get("spread")
        
        logger.log("info", f"Arbitrage opportunity detected for {asset} with spread {spread * 100:.2f}%")

        # Example: Fetch market data based on opportunity and trigger bot action
        action = rl_agent.decide_action(opportunity)
        
        # Check safety and execute trade
        if safety_manager.check_safety(opportunity):
            asyncio.create_task(trade_executor.execute_trade(action))  # Non-blocking
        else:
            logger.log("warning", "Safety conditions not met. Skipping trade execution.")
    # ---------------- Opportunity Handler Ends Here ----------------
    
    async def start_arbitrage_bots(self, strategies):
        """
        Asynchronously start the arbitrage bots based on the provided strategies.
        """
        logger.log("info", "Starting arbitrage bots...")
        tasks = []

        for strategy in strategies:
            if self.is_valid_strategy(strategy):
                tasks.append(self.run_arbitrage_bot(strategy))

        self.active_tasks = await asyncio.gather(*tasks)

    async def run_arbitrage_bot(self, strategy):
        """
        Run a single arbitrage bot based on a given strategy, ensuring all safety checks.
        """
        try:
            logger.log("info", f"Running arbitrage bot for strategy: {strategy.name}")

            while True:
                # Fetch market data and perform AI-driven decision making
                market_data = await strategy.fetch_market_data()
                logger.log("info", f"Fetched market data: {market_data}")

                action = rl_agent.decide_action(market_data)
                logger.log("info", f"RL Decision for arbitrage: {action}")

                # Safety checks before executing trade
                if safety_manager.check_safety(market_data):
                    await trade_executor.execute_trade(action)
                else:
                    logger.log("warning", "Safety conditions not met. Skipping trade execution.")

                await asyncio.sleep(strategy.interval)  # Re-run based on the strategy interval

        except Exception as e:
            logger.log("error", f"Error in arbitrage bot for strategy {strategy.name}: {str(e)}")
            handle_errors(e)

    def is_valid_strategy(self, strategy):
        """
        Validate the strategy based on various lists (e.g., whitelist, blacklist, etc.).
        """
        if list_manager.is_blacklisted(strategy):
            logger.log("warning", f"Strategy {strategy.name} is blacklisted. Skipping.")
            return False
        return True

    async def stop_arbitrage_bots(self):
        """
        Stop all running arbitrage bots gracefully.
        """
        logger.log("info", "Stopping arbitrage bots...")
        for task in self.active_tasks:
            task.cancel()
        await asyncio.gather(*self.active_tasks, return_exceptions=True)
        self.active_tasks.clear()
