# Full file path: /moneyverse/strategies/arbitrage_bot.py

import asyncio
from ai.agents.rl_agent import RLTradingAgent
from centralized_logger import CentralizedLogger
from src.managers.transaction_manager import TransactionManager
from src.managers.risk_manager import RiskManager
from market_data import MarketDataAPI
from src.utils.error_handler import handle_errors
from ai.rl_algorithms import DDPGAgent, PPOAgent, DQNAgent, MARLAgent, PGMPolicyGradient, LSTM_MemoryAgent

# Initialize components
logger = CentralizedLogger()
transaction_manager = TransactionManager()
risk_manager = RiskManager()
market_data_api = MarketDataAPI()

# Initialize multiple RL agents, including memory-based LSTM agent
ddpg_agent = DDPGAgent(environment="arbitrage_trading")
ppo_agent = PPOAgent(environment="arbitrage_trading")
dqn_agent = DQNAgent(environment="arbitrage_trading")
marl_agent = MARLAgent(environment="arbitrage_trading")
pgm_agent = PGMPolicyGradient(environment="arbitrage_trading")
lstm_agent = LSTM_MemoryAgent(environment="arbitrage_trading")  # Memory-based agent with LSTM

# Track performance for each agent
agent_performance = {
    "DDPG": 0,
    "PPO": 0,
    "DQN": 0,
    "MARL": 0,
    "PGM": 0,
    "LSTM": 0  # Include performance tracking for LSTM agent
}

class ArbitrageBot:
    def __init__(self):
        self.running = True
        self.rl_agent = None
        self.select_agent()  # Initialize with the best performing agent

    def select_agent(self):
        """Select the agent with the highest recent performance, adapting to market changes."""
        best_agent_name = max(agent_performance, key=agent_performance.get)
        self.rl_agent = {
            "DDPG": ddpg_agent,
            "PPO": ppo_agent,
            "DQN": dqn_agent,
            "MARL": marl_agent,
            "PGM": pgm_agent,
            "LSTM": lstm_agent  # Include LSTM memory agent for decision-making
        }[best_agent_name]
        logger.log_info(f"Selected agent: {best_agent_name}")

    async def fetch_market_data(self):
        """Fetch market data for arbitrage opportunities."""
        try:
            market_data = await market_data_api.get_arbitrage_data()
            logger.log_info(f"Fetched market data: {market_data}")
            return market_data
        except Exception as e:
            logger.log_error(f"Failed to fetch market data: {e}")
            handle_errors(e)
            return None

    def ai_decision(self, market_data):
        """Make an AI decision using the selected RL agent, potentially utilizing memory (LSTM)."""
        try:
            action = self.rl_agent.decide_action(market_data)
            logger.log_info(f"AI decision by {self.rl_agent.__class__.__name__}: {action}")
            return action
        except Exception as e:
            logger.log_error(f"Error in AI decision-making: {e}")
            handle_errors(e)
            return None

    async def execute_trade(self, trade_data):
        """Execute the trade and update agent performance based on success or failure."""
        trade_success = await transaction_manager.execute_trade(trade_data)
        agent_type = self.rl_agent.__class__.__name__

        if trade_success:
            logger.log_info(f"Trade executed successfully: {trade_data}")
            agent_performance[agent_type] += 1  # Reward success
        else:
            logger.log_warning(f"Trade failed: {trade_data}")
            agent_performance[agent_type] -= 1  # Penalize failure

    async def run(self):
        """Main loop to operate the arbitrage bot with dynamic agent switching based on performance."""
        logger.log_info("Starting Arbitrage Bot with dynamic agent selection and memory-based reinforcement learning...")
        try:
            while self.running:
                # Evaluate agent performance periodically to potentially switch agents
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
                        "trade_type": "arbitrage"
                    }
                    await self.execute_trade(trade_data)
                else:
                    logger.log_warning("Risk thresholds exceeded. Skipping trade execution.")

                await asyncio.sleep(30)  # Adjust as needed for desired frequency

        except Exception as e:
            logger.log_error(f"Critical error in Arbitrage Bot: {e}")
            handle_errors(e)

    def stop(self):
        """Stops the arbitrage bot."""
        logger.log_info("Stopping Arbitrage Bot...")
        self.running = False


if __name__ == "__main__":
    bot = ArbitrageBot()
    asyncio.run(bot.run())
