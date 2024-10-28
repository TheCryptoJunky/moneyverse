import asyncio
from ai.agents.rl_agent import RLTradingAgent
from centralized_logger import CentralizedLogger
from src.managers.transaction_manager import TransactionManager
from src.managers.risk_manager import RiskManager
from market_data import MarketDataAPI
from src.utils.error_handler import handle_errors

# Initialize components
logger = CentralizedLogger()
rl_agent = RLTradingAgent(environment="liquidity_drain")
transaction_manager = TransactionManager()
risk_manager = RiskManager()
market_data_api = MarketDataAPI()

class LiquidityDrainBot:
    def __init__(self):
        self.running = True

    async def run(self):
        logger.log_info("Starting Liquidity Drain Bot...")

        try:
            while self.running:
                # Step 1: Fetch market data
                market_data = await market_data_api.get_liquidity_drain_data()
                logger.log_info(f"Fetched liquidity drain market data: {market_data}")

                # Step 2: AI decision making
                action = rl_agent.decide_action(market_data)
                logger.log_info(f"RL action decision: {action}")

                # Step 3: Risk checks
                if risk_manager.is_risk_compliant(market_data):
                    trade_data = {
                        "source_wallet": action["source_wallet"],
                        "amount": action["amount"],
                        "trade_type": "liquidity_drain"
                    }

                    # Step 4: Execute trade
                    trade_success = await transaction_manager.execute_trade(trade_data)
                    if trade_success:
                        logger.log_info(f"Trade executed: {trade_data}")
                    else:
                        logger.log_warning(f"Trade failed: {trade_data}")
                else:
                    logger.log_warning("Risk thresholds exceeded. Skipping trade execution.")

                await asyncio.sleep(60)

        except Exception as e:
            logger.log_error(f"Error in Liquidity Drain Bot: {str(e)}")
            handle_errors(e)

    def stop(self):
        logger.log_info("Stopping Liquidity Drain Bot...")
        self.running = False

if __name__ == "__main__":
    bot = LiquidityDrainBot()
    asyncio.run(bot.run())
