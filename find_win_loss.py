import numpy as np
import pandas as pd

df = pd.read_csv('df.csv')

df['result'] = 'hold'

take_profit_condition = (df['cross'] == 1)


for i in range(len(df)):
    if df['cross'][i] == 1 :
        take_profit_price = df['close'][i] + 0.00100
        stoploss_price = df['close'][i] - 0.00050
        for j in range(1, 1000) :
            if i + j >= 99798 :
                break
            if df['high'][i + j] >= take_profit_price :
                df.loc[i, 'result'] = 'win'
                break
            elif df['high'][i + j] <= stoploss_price :
                df.loc[i, 'result'] = 'loss'
                break
    if df['cross'][i] == 2 :
        take_profit_price = df['close'][i] - 0.00100
        stoploss_price = df['close'][i] + 0.00050
        for j in range(1, 1000) :
            if i + j >= 99798 :
                break
            if df['high'][i + j] <= take_profit_price :
                df.loc[i, 'result'] = 'win'
                break
            elif df['high'][i + j] >= stoploss_price :
                df.loc[i, 'result'] = 'loss'
                break
                
condition_buy = df['cross'] == 1
df_buy = df[condition_buy]

condition_sell = df['cross'] == 2
df_sell = df[condition_sell]

df_buy.to_csv('df_buy.csv')
df_sell.to_csv('df_sell.csv')