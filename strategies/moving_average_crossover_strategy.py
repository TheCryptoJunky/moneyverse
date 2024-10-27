# /strategies/moving_average_crossover_strategy.py

import pandas as pd
import numpy as np

class MovingAverageCrossoverStrategy:
    def __init__(self, short_window, long_window):
        """
        Initialize the Moving Average Crossover Strategy.

        Parameters:
        short_window (int): The short window size for the moving average.
        long_window (int): The long window size for the moving average.
        """
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        """
        Generate buy and sell signals based on the moving average crossover strategy.

        Parameters:
        data (pd.DataFrame): The historical price data.

        Returns:
        pd.DataFrame: A DataFrame containing the buy and sell signals.
        """
        # Calculate moving averages
        short_ma = data['close'].rolling(window=self.short_window).mean()
        long_ma = data['close'].rolling(window=self.long_window).mean()

        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['buy'] = np.where((short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1)), 1, 0)
        signals['sell'] = np.where((short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1)), 1, 0)

        return signals

    def get_strategy_name(self):
        """
        Get the name of the strategy.

        Returns:
        str: The name of the strategy.
        """
        return "Moving Average Crossover Strategy"
