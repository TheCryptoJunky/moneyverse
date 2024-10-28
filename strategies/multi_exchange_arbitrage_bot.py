import asyncio
from ai.agents.rl_agent import RLTradingAgent
from centralized_logger import CentralizedLogger
from src.managers.transaction_manager import TransactionManager
from src.managers.risk_manager import RiskManager
from market_data import MarketDataAPI
from src.utils.error_handler import handle_errors

# Initialize components
logger = CentralizedLogger()
rl_agent = RLTradingAgent(environment="multi_exchange_arbitrage")
transaction_manager = TransactionManager()
risk_manager = RiskManager()
market_data_api = MarketDataAPI()

class MultiExchangeArbitrageBot:
    def __init__(self):
        self.running = True

    async def run(self):
        """
        Main loop for running the multi-exchange arbitrage bot.
        """
        logger.log_info("Starting Multi-Exchange Arbitrage Bot...")

        try:
            while self.running:
                # Step 1: Fetch market data from multiple exchanges
                market_data = await market_data_api.get_multi_exchange_arbitrage_data()
                logger.log_info(f"Fetched multi-exchange arbitrage market data: {market_data}")

                # Step 2: AI-driven decision making for arbitrage opportunities
                action = rl_agent.decide_action(market_data)
                logger.log_info(f"RL action decision: {action}")

                # Step 3: Risk checks before executing trades
                if risk_manager.is_risk_compliant(market_data):
                    trade_data = {
                        "source_wallet": action["source_wallet"],
                        "amount": action["amount"],
                        "trade_type": "multi_exchange_arbitrage"
                    }

                    # Step 4: Execute trade using the Transaction Manager
                    trade_success = await transaction_manager.execute_trade(trade_data)
                    if trade_success:
                        logger.log_info(f"Trade executed: {trade_data}")
                    else:
                        logger.log_warning(f"Trade failed: {trade_data}")
                else:
                    logger.log_warning("Risk thresholds exceeded. Skipping trade execution.")

                await asyncio.sleep(30)  # Adjust sleep time based on the market conditions

        except Exception as e:
            logger.log_error(f"Error in Multi-Exchange Arbitrage Bot: {str(e)}")
            handle_errors(e)

    def stop(self):
        """
        Gracefully stop the bot.
        """
        logger.log_info("Stopping Multi-Exchange Arbitrage Bot...")
        self.running = False

if __name__ == "__main__":
    bot = MultiExchangeArbitrageBot()
    asyncio.run(bot.run())
