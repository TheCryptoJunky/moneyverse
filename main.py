import asyncio
from strategies.mev_strategy import MEV_STRATEGIES
from managers.wallet_swarm import WalletSwarm
from utils.error_handler import log_error

# Initialize WalletSwarm and other required components
wallet_swarm = WalletSwarm()

class StrategyExecutor:
    def __init__(self, strategy_name, wallet_swarm):
        self.strategy_name = strategy_name
        self.wallet_swarm = wallet_swarm
        self.strategy = self.load_strategy()

    def load_strategy(self):
        """Dynamically load and initialize the selected strategy."""
        if self.strategy_name in MEV_STRATEGIES:
            return MEV_STRATEGIES[self.strategy_name](self.wallet_swarm)
        else:
            raise ValueError(f"Strategy '{self.strategy_name}' not found in MEV_STRATEGIES")

    async def execute_strategy(self, market_data):
        """Execute the loaded strategy by identifying opportunities and executing trades."""
        try:
            opportunities = self.strategy.identify_opportunities(market_data)
            if opportunities:
                for opportunity in opportunities:
                    profit = self.strategy.execute(opportunity)
                    print(f"Executed trade with expected profit: {profit}")
            else:
                print("No profitable opportunities identified.")
        except Exception as e:
            log_error(f"Error executing strategy {self.strategy_name}: {e}")

async def main():
    # Prompt user for strategy selection
    strategy_name = input("Enter the strategy to execute: ")
    strategy_executor = StrategyExecutor(strategy_name, wallet_swarm)

    # Placeholder for market data (replace with actual data in production)
    market_data = {
        'exchange1': 5000,
        'exchange2': 5100,
        'exchange3': 5050
    }

    try:
        # Run the selected strategy
        await strategy_executor.execute_strategy(market_data)
    except ValueError as e:
        log_error(f"Invalid strategy selection: {e}")
    except Exception as e:
        log_error(f"Unexpected error in main execution: {e}")

# Run the main loop
if __name__ == "__main__":
    asyncio.run(main())
