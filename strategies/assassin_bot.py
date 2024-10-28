# Full file path: /moneyverse/strategies/assassin_bot.py

import asyncio
from ai.agents.rl_agent import RLTradingAgent
from centralized_logger import CentralizedLogger
from src.managers.transaction_manager import TransactionManager
from src.managers.risk_manager import RiskManager
from market_data import MarketDataAPI
from src.utils.error_handler import handle_errors
from ai.rl_algorithms import DDPGAgent, PPOAgent, LSTM_MemoryAgent, MARLAgent

# Initialize components
logger = CentralizedLogger()
transaction_manager = TransactionManager()
risk_manager = RiskManager()
market_data_api = MarketDataAPI()

# Initialize multiple RL agents for dynamic selection, including memory-based LSTM
ddpg_agent = DDPGAgent(environment="assassin_trading")
ppo_agent = PPOAgent(environment="assassin_trading")
lstm_agent = LSTM_MemoryAgent(environment="assassin_trading")
marl_agent = MARLAgent(environment="assassin_trading")

# Track performance for each agent
agent_performance = {"DDPG": 0, "PPO": 0, "LSTM": 0, "MARL": 0}

class AssassinBot:
    def __init__(self):
        self.running = True
        self.rl_agent = None
        self.select_agent()  # Initialize with best performing agent

    def select_agent(self):
        """Select the best-performing agent based on recent trade performance."""
        best_agent_name = max(agent_performance, key=agent_performance.get)
        self.rl_agent = {
            "DDPG": ddpg_agent,
            "PPO": ppo_agent,
            "LSTM": lstm_agent,
            "MARL": marl_agent,
        }[best_agent_name]
        logger.log_info(f"Selected agent: {best_agent_name}")

    async def fetch_market_data(self):
        """Fetch market data for assassin trading opportunities."""
        try:
            market_data = await market_data_api.get_assassin_data()
            logger.log_info(f"Fetched market data: {market_data}")
            return market_data
        except Exception as e:
            logger.log_error(f"Failed to fetch market data: {e}")
            handle_errors(e)
            return None

    def ai_decision(self, market_data):
        """Make an AI-driven trading decision using the selected RL agent."""
        try:
            action = self.rl_agent.decide_action(market_data)
            logger.log_info(f"AI decision by {self.rl_agent.__class__.__name__}: {action}")
            return action
        except Exception as e:
            logger.log_error(f"Error in AI decision-making: {e}")
            handle_errors(e)
            return None

    async def execute_trade(self, trade_data):
        """Execute trade and update agent performance based on success or failure."""
        trade_success = await transaction_manager.execute_trade(trade_data)
        agent_type = self.rl_agent.__class__.__name__

        if trade_success:
            logger.log_info(f"Trade executed successfully: {trade_data}")
            agent_performance[agent_type] += 1  # Reward successful trade
        else:
            logger.log_warning(f"Trade failed: {trade_data}")
            agent_performance[agent_type] -= 1  # Penalize failed trade

    async def run(self):
        """Main loop to operate the assassin bot with dynamic agent switching."""
        logger.log_info("Starting Assassin Bot with dynamic agent selection...")
        try:
            while self.running:
                # Re-evaluate and select best-performing agent
                if max(agent_performance.values()) != agent_performance[self.rl_agent.__class__.__name__]:
                    self.select_agent()

                market_data = await self.fetch_market_data()
                if market_data is None:
                    logger.log_warning("No market data available; retrying in 30 seconds.")
                    await asyncio.sleep(30)
                    continue

                action = self.ai_decision(market_data)
                if action is None:
                    logger.log_warning("AI decision failed; retrying in 30 seconds.")
                    await asyncio.sleep(30)
                    continue

                if risk_manager.is_risk_compliant(market_data):
                    trade_data = {
                        "source_wallet": action.get("source_wallet"),
                        "amount": action.get("amount"),
                        "trade_type": "assassin"
                    }
                    await self.execute_trade(trade_data)
                else:
                    logger.log_warning("Risk thresholds exceeded. Skipping trade execution.")

                await asyncio.sleep(30)  # Adjust as needed for desired frequency

        except Exception as e:
            logger.log_error(f"Critical error in Assassin Bot: {e}")
            handle_errors(e)

    def stop(self):
        """Stops the assassin bot."""
        logger.log_info("Stopping Assassin Bot...")
        self.running = False

if __name__ == "__main__":
    bot = AssassinBot()
    asyncio.run(bot.run())
