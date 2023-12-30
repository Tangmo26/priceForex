import numpy as np
import pandas as pd

indicators = ['RSI', 'CCI', 'MACD', 'Signal', 'DIplus', 'DIminus', 'ADX', 'MFI', 'BBr', 'cumdelta_rsi', 'ROC', 'stoch_rsi_k', 'stoch_rsi_d', 'ParabolicSAR_Signal', 'ATR', 'DifEma50', 'DifEma200']
columnPrices =['high', 'low', 'close', 'open']

# Specify the number of previous columns you want to create
num_previous_columns = 50

def findBuySell(df, stoploss, takeprofit) :
    df["DifEma50"] = np.round(df['close'] - df['EMA50'], decimals= 5)
    df["DifEma200"] = np.round(df['close'] - df['EMA200'], decimals= 5)
    
    def find_crossovers(df, macd_col='macd', signal_col='signal', cross_col='cross'):
        df[cross_col] = 0  # Initialize the 'cross' column with zeros

        # Find crossover conditions
        crossover_condition = (df[macd_col] > df[signal_col]) & (df[macd_col].shift(1) <= df[signal_col].shift(1))
        crossunder_condition = (df[macd_col] < df[signal_col]) & (df[macd_col].shift(1) >= df[signal_col].shift(1))

        # Assign values to the 'cross' column based on conditions
        df.loc[crossover_condition, cross_col] = 1
        df.loc[crossunder_condition, cross_col] = 2

        return df
    
    df = find_crossovers(df, macd_col='MACD', signal_col='Signal', cross_col='cross')
    # Create previous columns for each indicator
    for indicator in indicators:
        for i in range(1, num_previous_columns + 1):
            prev_col_name = f'dif_{indicator}_pre{i}'
            df[prev_col_name] = np.round(df[indicator] - df[indicator].shift(i), decimals = 5)
            
    for columnPrice in columnPrices:
        for i in range(0, num_previous_columns + 1):
            prev_col_name = f'dif_{columnPrice}_pre{i}'
            df[prev_col_name] = np.round(df['close'] - df[columnPrice].shift(i), decimals = 5)
    
    df = df.iloc[num_previous_columns:].reset_index(drop=True)
            
    df['result'] = 'hold'

    for i in range(len(df)):
        if df['cross'][i] == 1 :
            take_profit_price = df['close'][i] + takeprofit
            stoploss_price = df['close'][i] - stoploss
            for j in range(1, 1000) :
                if i + j >= len(df['result']) :
                    break
                if df['high'][i + j] >= take_profit_price :
                    df.loc[i, 'result'] = 'win'
                    break
                elif df['low'][i + j] <= stoploss_price :
                    df.loc[i, 'result'] = 'loss'
                    break
        if df['cross'][i] == 2 :
            take_profit_price = df['close'][i] - takeprofit
            stoploss_price = df['close'][i] + stoploss
            for j in range(1, 1000) :
                if i + j >= len(df['result']) :
                    break
                if df['low'][i + j] <= take_profit_price :
                    df.loc[i, 'result'] = 'win'
                    break
                elif df['high'][i + j] >= stoploss_price :
                    df.loc[i, 'result'] = 'loss'
                    break
            
    df_buy = df[(df['cross'] == 1) & (df['result'] != 'hold')]
    df_sell = df[(df['cross'] == 2) & (df['result'] != 'hold')]
	
    return df_buy, df_sell
    