# moneyverse/managers/central_opportunity_manager.py

import logging
import asyncio
from strategies.volatility_arbitrage_bot import VolatilityArbitrageBot
from strategies.triangle_arbitrage_bot import TriangleArbitrageBot
from strategies.statistical_arbitrage_bot import StatisticalArbitrageBot
from strategies.yield_arbitrage_bot import YieldArbitrageBot
from strategies.spatial_arbitrage_bot import SpatialArbitrageBot
from strategies.intra_exchange_arbitrage import IntraExchangeArbitrageBot
from strategies.trade_batch_Flash_loan_arbitrage_bot import TradeBatchFlashLoanArbitrageBot
from helper_bots.mempool_monitor import MempoolMonitor
from helper_bots.flash_loan_monitor import FlashLoanMonitor

class CentralOpportunityManager:
    """
    Manages arbitrage and trading opportunities by evaluating and assigning them to the most suitable bot.
    Prevents competition among bots and optimizes resource allocation to meet core NAV goals.

    Attributes:
    - bots (dict): Dictionary of initialized strategy bots.
    - mempool_monitor (MempoolMonitor): Monitors the mempool for opportunities.
    - flash_loan_monitor (FlashLoanMonitor): Monitors flash loan providers for loan opportunities.
    - opportunity_queue (list): Queue of pending opportunities for evaluation.
    - logger (Logger): Logs manager actions and allocations.
    """

    def __init__(self, yield_monitor, trade_executor):
        # Initialize strategy bots with relevant monitors or executors as needed
        self.bots = {
            "volatility_arbitrage": VolatilityArbitrageBot(...),
            "triangle_arbitrage": TriangleArbitrageBot(...),
            "statistical_arbitrage": StatisticalArbitrageBot(...),
            "yield_arbitrage": YieldArbitrageBot(yield_monitor, trade_executor),
            "spatial_arbitrage": SpatialArbitrageBot(...),
            "intra_exchange_arbitrage": IntraExchangeArbitrageBot(...),
            "batch_flash_loan_arbitrage": TradeBatchFlashLoanArbitrageBot(...),
        }
        # Initialize helper bots for real-time opportunity detection
        self.mempool_monitor = MempoolMonitor()
        self.flash_loan_monitor = FlashLoanMonitor()
        self.opportunity_queue = []  # Queue to store detected opportunities for evaluation
        self.logger = logging.getLogger(__name__)
        self.logger.info("CentralOpportunityManager initialized.")

        # Set up helper bots with the opportunity callback
        self.setup_helper_bots()

    def setup_helper_bots(self):
        """
        Configures the helper bots with callbacks to feed detected opportunities
        into the Central Opportunity Manager's queue.
        """
        self.mempool_monitor.set_opportunity_callback(self.add_opportunity)
        self.flash_loan_monitor.set_opportunity_callback(self.add_opportunity)
        self.logger.info("Helper bots configured with opportunity callbacks.")

    async def evaluate_opportunities(self):
        """
        Continuously evaluates queued opportunities and assigns them to the appropriate bot.
        Ensures that each opportunity is handled by the best-suited bot to maximize NAV growth.
        """
        self.logger.info("Evaluating opportunities in the queue.")
        while True:
            if self.opportunity_queue:
                opportunity = self.opportunity_queue.pop(0)  # Retrieve the next opportunity
                await self.allocate_opportunity(opportunity)
            await asyncio.sleep(0.1)  # Adjust frequency for efficiency

    async def allocate_opportunity(self, opportunity: dict):
        """
        Assigns an opportunity to the most suitable bot based on opportunity type and bot availability.

        Args:
        - opportunity (dict): Dictionary containing details about the detected opportunity.
        """
        asset = opportunity.get("asset")
        type_ = opportunity.get("type")
        amount = opportunity.get("amount", 0)

        # Select the appropriate bot based on opportunity type
        if type_ == "volatility":
            bot_name = "volatility_arbitrage"
        elif type_ == "triangle":
            bot_name = "triangle_arbitrage"
        elif type_ == "statistical":
            bot_name = "statistical_arbitrage"
        elif type_ == "yield":
            bot_name = "yield_arbitrage"
        elif type_ == "spatial":
            bot_name = "spatial_arbitrage"
        elif type_ == "intra_exchange":
            bot_name = "intra_exchange_arbitrage"
        elif type_ == "batch_flash_loan":
            bot_name = "batch_flash_loan_arbitrage"
        else:
            bot_name = None  # If no specific type matches

        # Allocate to bot if found in bot registry
        if bot_name and bot_name in self.bots:
            chosen_bot = self.bots[bot_name]
            self.logger.info(f"Allocating {opportunity} to {bot_name}")
            await chosen_bot.handle_opportunity(opportunity)
        else:
            self.logger.warning("No suitable bot found for the opportunity.")

    def add_opportunity(self, opportunity: dict):
        """
        Adds a new opportunity to the queue for evaluation by the manager.

        Args:
        - opportunity (dict): Opportunity data including type, asset, amount, etc.
        """
        self.opportunity_queue.append(opportunity)
        self.logger.info(f"Opportunity added to queue: {opportunity}")

    async def start_monitoring(self):
        """
        Starts the helper bots to monitor and detect new opportunities automatically.
        """
        self.logger.info("Starting helper bot monitors for opportunity detection.")
        await asyncio.gather(
            self.mempool_monitor.detect_opportunities(),
            self.flash_loan_monitor.detect_opportunities()
        )
