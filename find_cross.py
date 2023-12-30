import numpy as np
import pandas as pd

df = pd.read_csv('ohlc.csv')

def find_crossovers(df, macd_col='macd', signal_col='signal', cross_col='cross'):
    
    df[cross_col] = 0  # Initialize the 'cross' column with zeros

    # Find crossover conditions
    crossover_condition = (df[macd_col] > df[signal_col]) & (df[macd_col].shift(1) <= df[signal_col].shift(1))
    crossunder_condition = (df[macd_col] < df[signal_col]) & (df[macd_col].shift(1) >= df[signal_col].shift(1))

    # Assign values to the 'cross' column based on conditions
    df.loc[crossover_condition, cross_col] = 1
    df.loc[crossunder_condition, cross_col] = 2

    return df

# df = find_crossovers(df, macd_col='MACD', signal_col='Signal', cross_col='cross')

df["DifEma50"] = np.round(df['close'] - df['EMA50'], decimals= 5)
df["DifEma200"] = np.round(df['close'] - df['EMA200'], decimals= 5)

df.to_csv('df.csv', index = False)