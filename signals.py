# signals.py
import pandas as pd
import numpy as np


def generate_trading_signals(df: pd.DataFrame,
                             forecast: pd.DataFrame,
                             profit_threshold: float = 0.01,
                             rsi_overbought: float = 70,
                             rsi_oversold: float = 30) -> pd.DataFrame:
    """
    Generate buy/sell signals based on forecast and RSI.

    - Buy when forecasted price >= current_close * (1 + profit_threshold) OR RSI <= rsi_oversold
    - Sell when forecasted price <= current_close * (1 - profit_threshold) OR RSI >= rsi_overbought

    Returns a DataFrame with columns: date, current_close, forecast, RSI, signal (1=buy, -1=sell, 0=hold)
    """
    # Align current price and RSI with forecast dates
    # Assume df has DateTime index and columns ['Close', 'RSI']
    df = df.copy()
    forecast = forecast.copy()
    # Ensure forecast has 'ds' and 'yhat'
    forecast = forecast.set_index('ds')
    # Merge current DF and forecast
    merged = pd.DataFrame(index=forecast.index)
    merged['forecast'] = forecast['yhat']
    merged['close'] = df['Close'].reindex(merged.index, method='ffill')
    merged['rsi'] = df['RSI'].reindex(merged.index, method='ffill')
    # Determine signals
    conditions = []
    signals = []
    for idx, row in merged.iterrows():
        signal = 0
        if row['forecast'] >= row['close'] * (1 + profit_threshold) or row['rsi'] <= rsi_oversold:
            signal = 1
        elif row['forecast'] <= row['close'] * (1 - profit_threshold) or row['rsi'] >= rsi_overbought:
            signal = -1
        signals.append(signal)
    merged['signal'] = signals
    merged.reset_index(inplace=True)
    merged.rename(columns={'index': 'date'}, inplace=True)
    return merged


def backtest_signals(signals_df: pd.DataFrame) -> dict:
    """
    Backtest the generated signals.

    Assumes signals_df has columns ['date', 'close', 'signal'].
    Execute market-on-open trades:
    - Buy at next period open if signal=1
    - Sell (exit) at next period open if holding and signal=-1

    Returns performance metrics: total_return, num_trades, win_rate, max_drawdown
    """
    # Simplify: assume no partial positions, 1 unit per trade
    positions = []  # tuples of (entry_price, exit_price)
    position = None
    for i in range(len(signals_df)-1):
        sig = signals_df.loc[i, 'signal']
        open_price_next = signals_df.loc[i+1, 'close']  # approximate open ~ close
        if sig == 1 and position is None:
            # open long
            position = {'entry': open_price_next}
        elif sig == -1 and position is not None:
            # close long
            position['exit'] = open_price_next
            positions.append(position)
            position = None
    # close any open position at last price
    if position is not None:
        position['exit'] = signals_df.iloc[-1]['close']
        positions.append(position)
    # Compute returns
    returns = [(p['exit'] - p['entry']) / p['entry'] for p in positions]
    total_return = np.prod([1 + r for r in returns]) - 1 if returns else 0
    wins = [1 for r in returns if r > 0]
    win_rate = sum(wins)/len(returns) if returns else 0
    # Max drawdown: using equity curve
    equity = np.cumprod([1] + returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak)/peak
    max_dd = drawdowns.min()
    return {
        'total_return': total_return,
        'num_trades': len(returns),
        'win_rate': win_rate,
        'max_drawdown': max_dd
    }
