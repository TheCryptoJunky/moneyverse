# algorithms/__init__.py

# Import all strategy bot classes for access across the project
from strategies.arbitrage_bot import ArbitrageBot
from strategies.assassin_bot import AssassinBot
from strategies.cross_chain_arbitrage import CrossChainArbitrageBot
from strategies.enhanced_sandwich_attack_bot import EnhancedSandwichAttackBot
from strategies.flash_loan_arbitrage_bot import FlashLoanArbitrageBot
from strategies.front_running_bot import FrontRunningBot
from strategies.latency_arbitrage_bot import LatencyArbitrageBot
from strategies.liquidity_drain_bot import LiquidityDrainBot
from strategies.liquidity_provision_arbitrage_bot import LiquidityProvisionArbitrageBot
from strategies.market_impact_arbitrage_bot import MarketImpactArbitrageBot
from strategies.market_maker_bot import MarketMakerBot
from strategies.mev_strategy import MEVStrategy
from strategies.moving_average_crossover_strategy import MovingAverageCrossoverStrategy
from strategies.multi_exchange_arbitrage_bot import MultiExchangeArbitrageBot
from strategies.protection import ProtectionBot
from strategies.reinforced_accumulation_bot import ReinforcedAccumulationBot
from strategies.revenge_bot import RevengeBot
from strategies.sniper_bot import SniperBot
from strategies.statistical_arbitrage_bot import StatisticalArbitrageBot
from strategies.triangle_arbitrage import TriangleArbitrageBot

__all__ = [
    "ArbitrageBot",
    "AssassinBot",
    "CrossChainArbitrageBot",
    "EnhancedSandwichAttackBot",
    "FlashLoanArbitrageBot",
    "FrontRunningBot",
    "LatencyArbitrageBot",
    "LiquidityDrainBot",
    "LiquidityProvisionArbitrageBot",
    "MarketImpactArbitrageBot",
    "MarketMakerBot",
    "MEVStrategy",
    "MovingAverageCrossoverStrategy",
    "MultiExchangeArbitrageBot",
    "ProtectionBot",
    "ReinforcedAccumulationBot",
    "RevengeBot",
    "SniperBot",
    "StatisticalArbitrageBot",
    "TriangleArbitrageBot",
]
