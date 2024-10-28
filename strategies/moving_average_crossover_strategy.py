# Full file path: /moneyverse/strategies/moving_average_crossover_strategy.py

import pandas as pd
import numpy as np
from ai.rl_algorithms import PPOAgent, LSTM_MemoryAgent

class MovingAverageCrossoverStrategy:
    """
    Implements a dynamic Moving Average Crossover Strategy with RL-based adaptive window sizes.
    """
    def __init__(self, short_window=40, long_window=100):
        self.short_window = short_window
        self.long_window = long_window
        self.agent = PPOAgent(environment="moving_average_crossover")  # RL agent for adjusting parameters

    def generate_signals(self, data):
        """
        Generates buy/sell signals based on moving average crossover, adjusting windows based on RL agent.
        """
        # Update window sizes based on RL agent decisions
        self.adapt_window_sizes(data)

        short_ma = data['close'].rolling(window=self.short_window).mean()
        long_ma = data['close'].rolling(window=self.long_window).mean()

        signals = pd.DataFrame(index=data.index)
        signals['signal'] = np.where(short_ma > long_ma, 1, 0)  # Buy signal when short MA crosses above long MA

        return signals

    def adapt_window_sizes(self, data):
        """
        Uses the RL agent to adjust short and long window sizes based on recent performance.
        """
        state = self.get_state(data)
        action = self.agent.decide_action(state)

        # Assuming action output influences short and long window sizes
        self.short_window = max(20, int(self.short_window + action['short_adjustment']))
        self.long_window = max(self.short_window + 20, int(self.long_window + action['long_adjustment']))

    def get_state(self, data):
        """
        Constructs the current state for the RL agent, based on recent price trends and market volatility.
        """
        recent_data = data['close'][-self.long_window:]
        state = {
            "recent_trend": np.mean(recent_data.pct_change().fillna(0)),
            "volatility": np.std(recent_data.pct_change().fillna(0))
        }
        return state
