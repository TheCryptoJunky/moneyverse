import os

# Project settings
PROJECT_NAME = "Moneyverse"
PROJECT_VERSION = "1.0"

# API keys
API_KEY_BSCSCAN = os.environ.get("API_KEY_BSCSCAN")
API_KEY_ETHERSCAN = os.environ.get("API_KEY_ETHERSCAN")

# Wallet settings
WALLET_ADDRESS = os.environ.get("WALLET_ADDRESS")
WALLET_PRIVATE_KEY = os.environ.get("WALLET_PRIVATE_KEY")

# MEV strategy settings
MEV_STRATEGY = "simple"  # Options: simple, advanced
MEV_THRESHOLD = 0.5  # Threshold for MEV strategy

# Logging settings
LOG_LEVEL = "INFO"
LOG_FILE = "moneyverse.log"

class Config:
    risk_thresholds = {
        'stop_loss': -0.1,  # 10% stop loss
        'take_profit': 0.2,  # 20% take profit
        'max_trade_frequency': 10  # max 10 trades per hour
    }
    trade_executor_sleep_time = 60 # 60 seconds
