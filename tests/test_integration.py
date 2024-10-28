# Full file path: /moneyverse/tests/test_integration.py

import pytest
from strategies.mev_strategy import MEV_STRATEGIES
from main import StrategyExecutor
from src.managers.wallet_swarm import WalletSwarm

wallet_swarm = WalletSwarm()

@pytest.mark.parametrize("strategy_name", MEV_STRATEGIES.keys())
async def test_strategy_execution(strategy_name):
    """Test if each strategy in MEV_STRATEGIES can be initialized and executed."""
    strategy_executor = StrategyExecutor(strategy_name, wallet_swarm)
    market_data = {
        'exchange1': 5000,
        'exchange2': 5100,
        'exchange3': 5050
    }

    # Run the selected strategy
    await strategy_executor.execute_strategy(market_data)
    assert True, f"{strategy_name} executed successfully."
