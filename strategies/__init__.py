# strategies/__init__.py

# Import all strategy bot classes for streamlined access across the project
from .arbitrage_bot import ArbitrageBot
from .assassin_bot import AssassinBot
from .cross_chain_arbitrage import CrossChainArbitrageBot
from .enhanced_sandwich_attack_bot import EnhancedSandwichAttackBot
from .flash_loan_arbitrage_bot import FlashLoanArbitrageBot
from .front_running_bot import FrontRunningBot
from .latency_arbitrage_bot import LatencyArbitrageBot
from .liquidity_drain_bot import LiquidityDrainBot
from .liquidity_provision_arbitrage_bot import LiquidityProvisionArbitrageBot
from .market_impact_arbitrage_bot import MarketImpactArbitrageBot
from .market_maker_bot import MarketMakerBot
from .mev_strategy import MEVStrategy
from .moving_average_crossover_strategy import MovingAverageCrossoverStrategy
from .multi_exchange_arbitrage_bot import MultiExchangeArbitrageBot
from .protection import ProtectionBot
from .reinforced_accumulation_bot import ReinforcedAccumulationBot
from .revenge_bot import RevengeBot
from .sniper_bot import SniperBot
from .statistical_arbitrage_bot import StatisticalArbitrageBot
from .triangle_arbitrage import TriangleArbitrageBot

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
