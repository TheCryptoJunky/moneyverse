import asyncio
from ai.dnn_model import DNNModel
from ai_helpers import ReinvestmentHelper
from centralized_logger import CentralizedLogger
from profit_manager import ProfitManager
from src.safety.safety_manager import SafetyManager
from src.services.aggregator_service import AggregatorService
from utils.retry_decorator import retry
from self_learning.self_learning_engine import SelfLearningEngine

logger = CentralizedLogger()
dnn_model = DNNModel()
reinvestment_helper = ReinvestmentHelper()
safety_manager = SafetyManager()
aggregator_service = AggregatorService()
self_learning_engine = SelfLearningEngine()

class ProfitReinvestmentBot:
    """
    Profit Reinvestment Bot with dual-mode operation:
    1. Scheduled reinvestment
    2. Real-time reinvestment based on signals from the Self-Learning Engine
    """

    def __init__(self, reinvestment_threshold=100, schedule_interval=3600):
        self.profit_manager = ProfitManager()
        self.reinvestment_threshold = reinvestment_threshold  # Min threshold for reinvestment
        self.schedule_interval = schedule_interval  # Reinvestment interval in seconds
        self.running = True  # Control variable for async loop

    @retry(retries=3, delay=2)
    def collect_profits(self):
        """Collect profits from various sources with retry logic."""
        return self.profit_manager.collect_profits()

    def generate_reinvestment_plan(self, profits):
        """
        Generate a reinvestment plan based on current profits and market conditions.
        """
        reinvestment_plan = dnn_model.generate_reinvestment_plan(profits)
        logger.log("info", f"Generated reinvestment plan: {reinvestment_plan}")
        return reinvestment_plan

    def select_best_aggregator(self, reinvestment_plan):
        """
        Select the most cost-effective aggregator for reinvesting assets.
        """
        aggregators = aggregator_service.get_available_aggregators()
        best_aggregator = min(aggregators, key=lambda agg: agg.get_trade_cost(reinvestment_plan))
        logger.log("info", f"Selected aggregator: {best_aggregator.name} for reinvestment.")
        return best_aggregator

    async def collect_and_reinvest_profits(self):
        """
        Collect profits and reinvest based on AI-driven strategies to maximize NAV growth.
        """
        logger.log("info", "Starting profit collection and reinvestment...")

        try:
            # Step 1: Collect profits
            profits = self.collect_profits()
            if profits < self.reinvestment_threshold:
                logger.log("info", "Profits below threshold. Skipping reinvestment.")
                return
            logger.log("info", f"Collected profits: {profits}")

            # Step 2: Generate reinvestment plan using AI model
            reinvestment_plan = self.generate_reinvestment_plan(profits)

            # Step 3: Safety check before reinvesting
            if safety_manager.check_safety(profits):
                # Step 4: Select cost-effective aggregator
                best_aggregator = self.select_best_aggregator(reinvestment_plan)

                # Step 5: Execute reinvestment
                reinvestment_helper.execute_reinvestment(
                    reinvestment_plan=reinvestment_plan,
                    aggregator=best_aggregator
                )
                logger.log("info", f"Reinvestment executed with {best_aggregator.name}.")
            else:
                logger.log("warning", "Safety conditions not met. Skipping reinvestment.")

        except Exception as e:
            logger.log("error", f"Error in reinvestment process: {str(e)}")

    async def reinvest_based_on_signal(self):
        """
        Real-time reinvestment triggered by signals from the Self-Learning Engine.
        """
        while self.running:
            await asyncio.sleep(1)  # Check for signals every second
            try:
                if self_learning_engine.should_reinvest():
                    logger.log("info", "Self-Learning Engine signal for reinvestment detected.")
                    await self.collect_and_reinvest_profits()
            except Exception as e:
                logger.log("error", f"Error during real-time reinvestment processing: {str(e)}")

    async def scheduled_reinvestment(self):
        """
        Scheduled reinvestment function for periodic reinvestment.
        """
        while self.running:
            try:
                logger.log("info", "Running scheduled reinvestment check...")
                await asyncio.sleep(self.schedule_interval)  # Wait for the defined interval
                await self.collect_and_reinvest_profits()
            except Exception as e:
                logger.log("error", f"Error during scheduled reinvestment: {str(e)}")

    async def run_bot(self):
        """
        Main async function to initiate both real-time and scheduled reinvestment.
        """
        await asyncio.gather(
            self.reinvest_based_on_signal(),
            self.scheduled_reinvestment()
        )

    def stop(self):
        """
        Stop the reinvestment bot gracefully.
        """
        self.running = False
        logger.log("info", "Profit Reinvestment Bot stopped.")

if __name__ == "__main__":
    bot = ProfitReinvestmentBot()
    try:
        asyncio.run(bot.run_bot())
    except KeyboardInterrupt:
        bot.stop()
