from ai.rl_agent import RLTradingAgent
from ai.models.sentiment_analysis import SentimentAnalysisModel
from centralized_logger import CentralizedLogger
from market_data import MarketDataAPI
from helper_bots.profit_reinvestment_bot import ProfitReinvestmentBot
from src.services.aggregator_service import AggregatorService
from src.services.gas_price_service import GasPriceService
from utils.retry_decorator import retry
import asyncio

# Centralized logger setup for tracking engine behavior
logger = CentralizedLogger()

class SelfLearningEngine:
    """
    Self-Learning Engine for achieving the primary NAV goal by dynamically managing assets
    between volatile and stable assets in a cost-effective manner. Includes periodic reinvestment.
    """

    def __init__(self, reinvestment_interval=3600):
        # Initialize essential AI modules and data services
        self.rl_agent = RLTradingAgent(environment="nav_optimization", model="PPO")
        self.sentiment_model = SentimentAnalysisModel()
        self.market_data = MarketDataAPI()
        self.aggregator_service = AggregatorService()
        self.gas_service = GasPriceService()
        self.profit_reinvestment_bot = ProfitReinvestmentBot()
        self.reinvestment_interval = reinvestment_interval  # Interval in seconds for reinvestment

    @retry(retries=3, delay=2)
    def fetch_market_data(self):
        """Fetch liquidity and price data with retry logic for robustness."""
        return self.market_data.get_all_data()

    async def predict_market_behavior(self, data):
        """
        Use multiple models to predict market trends, targeting NAV optimization.
        Integrates both Transformer and LSTM for accurate trend analysis.
        """
        lstm_prediction = self.rl_agent.lstm_model.predict(data)
        transformer_prediction = self.rl_agent.transformer_model.predict(data)
        combined_prediction = (0.5 * lstm_prediction) + (0.5 * transformer_prediction)
        logger.log("info", f"Predicted market trend (NAV focus): {combined_prediction}")
        return combined_prediction

    def select_aggregator(self, amount, asset_type):
        """
        Choose the most cost-effective aggregator for asset conversion.
        Selects based on asset type and amount to maximize NAV.
        """
        aggregators = self.aggregator_service.get_available_aggregators()
        best_aggregator = min(aggregators, key=lambda agg: agg.get_trade_cost(amount, asset_type))
        logger.log("info", f"Selected aggregator: {best_aggregator.name} for {asset_type}")
        return best_aggregator

    async def adjust_asset_allocation(self, volatility_score):
        """
        Adjusts asset allocation based on volatility to maximize NAV growth.
        High volatility favors volatile assets; low volatility favors stability.
        """
        target_asset = "volatile" if volatility_score > 0.7 else "stable"
        
        # Perform allocation adjustments through aggregator
        for wallet in self.wallet_swarms:
            amount = wallet["balance"]
            selected_aggregator = self.select_aggregator(amount, target_asset)
            self.aggregator_service.execute_swap(
                wallet_id=wallet["wallet_id"],
                target_asset=target_asset,
                amount=amount,
                aggregator=selected_aggregator
            )
            logger.log("info", f"Swapped assets in {wallet['wallet_id']} to {target_asset}")

    def decide_action_with_rl(self, data):
        """
        Determine actions using RL model, focusing on NAV-optimized strategies.
        """
        return self.rl_agent.decide_action(data)

    async def periodic_reinvestment(self):
        """
        Periodically triggers the profit reinvestment bot to maximize NAV growth.
        """
        while True:
            try:
                logger.log("info", "Initiating periodic reinvestment...")
                await self.profit_reinvestment_bot.collect_and_reinvest_profits()
                await asyncio.sleep(self.reinvestment_interval)
            except Exception as e:
                logger.log("error", f"Error during reinvestment trigger: {str(e)}")

    async def nav_optimization_loop(self):
        """
        Main loop for NAV optimization, integrating reinforcement learning actions.
        """
        while True:
            try:
                # Step 1: Fetch real-time market data
                market_data = self.fetch_market_data()
                logger.log("info", f"Market data: {market_data}")

                # Step 2: Predict market trends and adjust NAV
                market_trend = await self.predict_market_behavior(market_data)
                await self.adjust_asset_allocation(market_trend["volatility_score"])

                # Step 3: Take actions based on RL decision-making
                rl_action = self.decide_action_with_rl(market_data)
                if rl_action:
                    self.execute_action(rl_action)
                else:
                    logger.log("warning", "RL action not taken due to insufficient confidence.")

            except Exception as e:
                logger.log("error", f"Error in NAV optimization loop: {str(e)}")
            
            await asyncio.sleep(10)  # Delay for continuous NAV monitoring

    def execute_action(self, rl_action):
        """
        Execute trading or asset conversion based on the RL-decided action.
        """
        logger.log("info", f"Executing NAV-focused action: {rl_action}")
        # Placeholder for action execution logic

    async def run_engine(self):
        """
        Runs the Self-Learning Engine with both NAV optimization and reinvestment.
        """
        await asyncio.gather(
            self.nav_optimization_loop(),
            self.periodic_reinvestment()
        )

if __name__ == "__main__":
    engine = SelfLearningEngine()
    try:
        asyncio.run(engine.run_engine())
    except KeyboardInterrupt:
        logger.log("info", "Self-Learning Engine stopped.")
