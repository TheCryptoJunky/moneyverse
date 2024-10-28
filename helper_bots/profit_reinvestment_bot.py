from ai.dnn_model import DNNModel
from ai_helpers import ReinvestmentHelper
from centralized_logger import CentralizedLogger
from profit_manager import ProfitManager
from src.safety.safety_manager import SafetyManager

logger = CentralizedLogger()
dnn_model = DNNModel()
reinvestment_helper = ReinvestmentHelper()
safety_manager = SafetyManager()

class ProfitReinvestmentBot:
    def __init__(self):
        self.profit_manager = ProfitManager()

    def collect_and_reinvest_profits(self):
        """
        Collect profits from different bots and reinvest them using AI-driven strategies.
        """
        logger.log("info", "Collecting profits for reinvestment...")

        try:
            # Collect profits
            profits = self.profit_manager.collect_profits()
            logger.log("info", f"Collected profits: {profits}")

            # Use AI model to generate reinvestment strategies
            reinvestment_plan = dnn_model.generate_reinvestment_plan(profits)
            logger.log("info", f"AI Reinvestment Plan: {reinvestment_plan}")

            if safety_manager.check_safety(profits):
                reinvestment_helper.execute_reinvestment(reinvestment_plan)
            else:
                logger.log("warning", "Safety conditions not met. Skipping reinvestment.")

        except Exception as e:
            logger.log("error", f"Error in reinvestment: {str(e)}")

if __name__ == "__main__":
    bot = ProfitReinvestmentBot()
    bot.collect_and_reinvest_profits()
