# moneyverse/utils/monitor_integration_snapshot.py

from moneyverse.helper_bots.mempool_monitor import MempoolMonitor
from moneyverse.utils.mempool_analysis import analyze_mempool  # Assumed analysis function

# Instantiate MempoolMonitor with analysis function from mempool_analysis.py
mempool_monitor = MempoolMonitor(analysis_func=analyze_mempool)

# Function to integrate all managers and strategies in one call
def initialize_monitor_with_all():
    # Register managers with relevant coordination functions
    from moneyverse.managers.arbitrage_manager import handle_arbitrage_opportunity
    from moneyverse.managers.risk_manager import manage_risk_on_opportunity
    from moneyverse.managers.transaction_manager import manage_transactions_on_opportunity

    # Register managers
    mempool_monitor.register_manager(handle_arbitrage_opportunity)
    mempool_monitor.register_manager(manage_risk_on_opportunity)
    mempool_monitor.register_manager(manage_transactions_on_opportunity)

    # Register strategies with opportunity response functions
    from moneyverse.strategies.cross_chain_arbitrage import cross_chain_arbitrage_handler
    from moneyverse.strategies.enhanced_sandwich_attack_bot import sandwich_attack_handler
    from moneyverse.strategies.flash_loan_arbitrage_bot import flash_loan_handler

    # Register strategies
    mempool_monitor.register_strategy(cross_chain_arbitrage_handler)
    mempool_monitor.register_strategy(sandwich_attack_handler)
    mempool_monitor.register_strategy(flash_loan_handler)

    return mempool_monitor  # Returns fully initialized monitor
