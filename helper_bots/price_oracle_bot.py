from ai.lstm_model import LSTMModel
from ai.arima_model import ARIMAModel
from centralized_logger import CentralizedLogger
from market_data import MarketDataAPI
from src.safety.safety_manager import SafetyManager
from src.services.aggregator_service import AggregatorService

logger = CentralizedLogger()
lstm_model = LSTMModel()
arima_model = ARIMAModel()
safety_manager = SafetyManager()
aggregator_service = AggregatorService()

class PriceOracleBot:
    """
    Price Oracle Bot that fetches price data, predicts future prices, and executes arbitrage trades.
    This enhanced version integrates autonomous decision-making and real-time adjustments.
    """

    def __init__(self):
        self.market_data = MarketDataAPI()
        self.arbitrage_threshold = 0.02  # Arbitrage threshold in percentage for executing trades

    def fetch_and_predict_prices(self):
        """
        Fetch prices from exchanges and predict future prices using ensemble AI models.
        """
        logger.log("info", "Fetching prices from exchanges...")

        try:
            # Fetch real-time prices
            prices = self.market_data.get_prices()
            logger.log("info", f"Fetched prices: {prices}")

            # Ensemble prediction using LSTM and ARIMA models
            lstm_prediction = lstm_model.predict(prices)
            arima_prediction = arima_model.predict(prices)
            combined_prediction = (0.7 * lstm_prediction) + (0.3 * arima_prediction)
            logger.log("info", f"Combined Prediction: {combined_prediction}")

            # Safety check and arbitrage execution
            if safety_manager.check_safety(prices):
                self.execute_arbitrage(prices, combined_prediction)
            else:
                logger.log("warning", "Safety conditions not met. Skipping arbitrage.")

        except Exception as e:
            logger.log("error", f"Error in price prediction: {str(e)}")

    def execute_arbitrage(self, current_prices, predicted_prices):
        """
        Execute arbitrage trades based on price predictions using a cost-effective aggregator.
        """
        price_differential = self.calculate_price_differential(current_prices, predicted_prices)
        if price_differential >= self.arbitrage_threshold:
            logger.log("info", f"Detected arbitrage opportunity with differential: {price_differential}")
            aggregator = self.select_cost_effective_aggregator(current_prices, predicted_prices)
            if aggregator:
                # Execute trade through selected aggregator
                success = aggregator_service.execute_trade(
                    aggregator=aggregator,
                    amount=price_differential,
                    asset="selected_asset"  # Placeholder for the chosen asset
                )
                if success:
                    logger.log("info", f"Arbitrage executed via {aggregator.name}")
                else:
                    logger.log("error", f"Arbitrage execution failed with {aggregator.name}")
        else:
            logger.log("info", "No viable arbitrage opportunity detected.")

    def calculate_price_differential(self, current_prices, predicted_prices):
        """
        Calculate price differential between current and predicted prices.
        Returns a percentage indicating the arbitrage opportunity.
        """
        return abs((predicted_prices - current_prices) / current_prices)

    def select_cost_effective_aggregator(self, current_prices, predicted_prices):
        """
        Choose the lowest-cost aggregator for arbitrage based on current and predicted prices.
        """
        aggregators = aggregator_service.get_available_aggregators()
        best_aggregator = min(aggregators, key=lambda agg: agg.get_trade_cost("selected_asset"))
        logger.log("info", f"Selected aggregator: {best_aggregator.name} for cost-effective arbitrage.")
        return best_aggregator

if __name__ == "__main__":
    bot = PriceOracleBot()
    bot.fetch_and_predict_prices()
