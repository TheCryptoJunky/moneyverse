# /config/settings.py

import os

# Load configuration from environment variables or fall back to defaults
CONFIG = {
    'whitelist': os.getenv('WHITELIST_PATH', 'config/whitelist.json'),  # Path to whitelist JSON file
    'blacklist': os.getenv('BLACKLIST_PATH', 'config/blacklist.json'),  # Path to blacklist JSON file
    'strategies': os.getenv('STRATEGIES', 'arbitrage,front_running,sandwich_attack').split(','),  # List of strategies
    'prometheus_config': os.getenv('PROMETHEUS_CONFIG', 'monitoring/prometheus_config.py'),  # Path to Prometheus config
    'gui_config': os.getenv('GUI_CONFIG', 'gui/frontend.py')  # Path to GUI configuration
}

# Example usage of dynamic strategies:
def load_strategies():
    """Dynamically load trading strategies based on environment configuration."""
    return CONFIG['strategies']
