# moneyverse/config/config.py

import os

# Trading settings
TRADE_INTERVAL = 60  # 1 minute
TRADE_PAIR = 'ETH-USDT'
TRADE_QUANTITY = 100

# Wallet swarm settings
WALLET_SWARM_SIZE = 10
WALLET_SWARM_DIVERSIFICATION = 0.5

# MEV strategy settings
MEV_STRATEGY = 'moving_average_crossover'
MEV_SHORT_WINDOW = 10
MEV_LONG_WINDOW = 30

# Risk management settings
RISK_MANAGER_ENABLED = True
RISK_MANAGER_THRESHOLD = 0.05

# RL agent settings
RL_AGENT_ENABLED = True
RL_AGENT_TRAINING_INTERVAL = 3600  # 1 hour

# Trade executor settings
TRADE_EXECUTOR_ENABLED = True
TRADE_EXECUTOR_TIMEOUT = 30

# API keys
API_KEY = 'YOUR_API_KEY'
API_SECRET = 'YOUR_API_SECRET'

# Exchange settings
EXCHANGE_NAME = 'binance'
EXCHANGE_API_URL = 'https://api.binance.com/api/v3'
