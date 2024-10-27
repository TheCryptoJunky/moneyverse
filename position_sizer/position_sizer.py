# position_sizer.py

import logging
from moneyverse.config import CONFIG
from moneyverse.risk_manager import manage_risk
import pandas as pd
import ccxt

def calculate_position_size(signal, risk_management_params):
    """
    Calculate the optimal position size for a given signal.

    Args:
        signal (dict): The trade signal containing the asset, direction, and other relevant information.
        risk_management_params (dict): The risk management parameters, including the maximum position size and stop-loss.

    Returns:
        float: The optimal position size.
    """
    # Define the exchange and the symbol
    exchange = ccxt.binance()
    symbol = signal["asset"]

    # Fetch the latest 14 days of 1-hour candlestick data
    bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=14 * 24)

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Calculate the Average True Range (ATR)
    df['hl'] = df['high'] - df['low']
    df['hc'] = abs(df['high'] - df['close'].shift(1))
    df['lc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()

    # Calculate the optimal position size based on the ATR and risk management parameters
    atr = df['atr'].iloc[-1]
    max_position_size = risk_management_params["max_position_size"]
    stop_loss = risk_management_params["stop_loss"]
    position_size = (max_position_size / atr) * (stop_loss / 100)

    return position_size

def main():
    logging.info("Position Sizer started")
    # Load the risk management parameters from the config file
    risk_management_params = CONFIG["risk_management"]

    # Example usage:
    signal = {"asset": "BTCUSDT", "direction": "long", "confidence": 0.8}
    position_size = calculate_position_size(signal, risk_management_params)
    logging.info(f"Optimal position size: {position_size}")

if __name__ == "__main__":
    main()
