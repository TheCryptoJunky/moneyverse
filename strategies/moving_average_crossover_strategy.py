# moneyverse/strategies/moving_average_crossover_strategy.py

import pandas as pd
import numpy as np

class MovingAverageCrossoverStrategy:
    def __init__(self, short_window, long_window):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        short_ma = data['close'].rolling(window=self.short_window).mean()
        long_ma = data['close'].rolling(window=self.long_window).mean()

        signals = pd.DataFrame(index=data.index)
        signals['signal'] = np.where(short_ma > long_ma, 1, 0)

        return signals
