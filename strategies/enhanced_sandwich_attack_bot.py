# Full file path: /moneyverse/strategies/enhanced_sandwich_attack_bot.py

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
ddpg_agent = DDPGAgent(environment="enhanced_sandwich_attack")
ppo_agent = PPOAgent(environment="enhanced_sandwich_attack")
lstm_agent = LSTM_MemoryAgent(environment="enhanced_sandwich_attack")
marl_agent = MARLAgent(environment="enhanced_sandwich_attack")

# Track performance for each agent
agent_performance = {"DDPG": 0, "PPO": 0, "LSTM": 0, "MARL": 0}

class EnhancedSandwichAttackBot:
    def __init__(self):
        self.running = True
        self.rl_agent = None
        self.select_agent()  # Initialize with the best performing agent

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
        """Fetch market data for sandwich attack opportunities."""
        try:
            market_data = await market_data_api.get_sandwich_attack_data()
            logger.log_info(f"Fetched sandwich attack market data: {market_data}")
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
        """Main loop to operate the enhanced sandwich attack bot with dynamic agent switching."""
        logger.log_info("Starting Enhanced Sandwich Attack Bot with dynamic agent selection...")
        try:
            while self.running:
                # Re-evaluate and select the best-performing agent if necessary
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
                        "trade_type": "sandwich_attack"
                    }
                    await self.execute_trade(trade_data)
                else:
                    logger.log_warning("Risk thresholds exceeded. Skipping trade execution.")

                await asyncio.sleep(30)  # Adjust based on desired trade frequency

        except Exception as e:
            logger.log_error(f"Critical error in Enhanced Sandwich Attack Bot: {e}")
            handle_errors(e)

    def stop(self):
        """Stops the enhanced sandwich attack bot."""
        logger.log_info("Stopping Enhanced Sandwich Attack Bot...")
        self.running = False

if __name__ == "__main__":
    bot = EnhancedSandwichAttackBot()
    asyncio.run(bot.run())
