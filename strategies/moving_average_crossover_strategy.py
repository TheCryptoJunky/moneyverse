# moneyverse/strategies/moving_average_crossover_strategy.py

import pandas as pd

class MovingAverageCrossoverStrategy:
    def __init__(self, symbol, timeframe, short_window=20, long_window=50):
        self.symbol = symbol
        self.timeframe = timeframe
        self.short_window = short_window
        self.long_window = long_window

    def calculate_signals(self, candles):
        # Calculate signals using a simple moving average crossover strategy
        candles['short_ma'] = candles['close'].rolling(window=self.short_window).mean()
        candles['long_ma'] = candles['close'].rolling(window=self.long_window).mean()

        signals = pd.Series(0, index=candles.index)
        signals[(candles['short_ma'] > candles['long_ma']) & (candles['close'] > candles['short_ma'])] = 1
        signals[(candles['short_ma'] < candles['long_ma']) & (candles['close'] < candles['short_ma'])] = -1

        return signals
