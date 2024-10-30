#__init__.py

# Import core components to facilitate access across modules
from .managers.wallet_manager import WalletManager
from .managers.goal_manager import GoalManager
from .managers.strategy_manager import StrategyManager
from .trade_executor import TradeExecutor
from .utils import error_handler, logging_util
from .flask_gui import create_app

__all__ = [
    "WalletManager",
    "GoalManager",
    "StrategyManager",
    "TradeExecutor",
    "error_handler",
    "logging_util",
    "create_app",
]
