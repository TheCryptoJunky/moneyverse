import logging
from moneyverse.config import Config
from moneyverse.mev_strategy import MEVStrategy
from moneyverse.wallet_swarm import WalletSwarm

class TradeExecutor:
    def __init__(self, mev_strategy: MEVStrategy, wallet_swarm: WalletSwarm, config: Config):
        self.mev_strategy = mev_strategy
        self.wallet_swarm = wallet_swarm
        self.config = config
        self.logger = logging.getLogger(__name__)

    def execute_trades(self, trades):
        """Execute a list of trades using the wallet swarm"""
        for trade in trades:
            try:
                self.wallet_swarm.execute_trade(trade)
                self.logger.info(f"Executed trade: {trade}")
            except Exception as e:
                self.logger.error(f"Failed to execute trade: {trade}, error: {e}")

    def start(self):
        """Get trades from the MEV strategy and execute them"""
        while True:
            trades = self.mev_strategy.get_trades()
            self.execute_trades(trades)
            # Sleep for a short period to avoid overwhelming the wallet swarm
            # This can be adjusted based on the performance of the wallet swarm
            import time
            time.sleep(self.config.trade_executor_sleep_time)

if __name__ == "__main__":
    # Example usage
    config = Config()
    mev_strategy = MEVStrategy(config)
    wallet_swarm = WalletSwarm(config)
    trade_executor = TradeExecutor(mev_strategy, wallet_swarm, config)
    trade_executor.start()
