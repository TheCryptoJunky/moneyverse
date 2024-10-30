# flask_gui/__init__.py

# Import main Flask GUI components for centralized access
from .app import create_app
from .dashboard import Dashboard
from .wallet_swarm_ui import WalletSwarmUI

__all__ = [
    "create_app",
    "Dashboard",
    "WalletSwarmUI",
]
