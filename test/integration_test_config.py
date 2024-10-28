import numpy as np
from src.ai.environment import TradingEnv

def create_test_config():
    """
    Create configuration for the integration test.
    This includes market data, agent configuration, and strategy parameters.
    """
    market_data = np.array([
        [100, 102, 105, 107],  # Example price series for asset 1
        [200, 198, 202, 205],  # Example price series for asset 2
        [300, 305, 308, 310]   # Example price series for asset 3
    ])
    
    agents_config = [
        {'market_data': market_data[0]},  # Agent 1 config for asset 1
        {'market_data': market_data[1]},  # Agent 2 config for asset 2
        {'market_data': market_data[2]},  # Agent 3 config for asset 3
    ]

    return agents_config
