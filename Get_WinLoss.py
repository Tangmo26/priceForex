import pandas as pd
import os
from find_win_loss import findBuySell

Curency = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCHF', 'USDCAD']
textTime = ['5m', '15m', '30m', '1h', '2h', '4h']
stoploss = [0.00050, 0.00100, 0.00250, 0.00400, 0.00500, 0.0600]

folder_path_buy = 'buyData'
folder_path_sell = 'sellData'
os.makedirs(folder_path_buy, exist_ok=True)
os.makedirs(folder_path_sell, exist_ok=True)

for curency in Curency :
    for j in range(len(textTime)) :
        df = []
        df = pd.read_csv(f'ohlc_AllCurency/ohlc_{curency}{textTime[j]}.csv')
        df_buy, df_sell = findBuySell(df, stoploss[j], stoploss[j] * 2)
        
        df_buy.to_csv(f'{folder_path_buy}/buy_{curency}{textTime[j]}.csv', index=False)
        df_sell.to_csv(f'{folder_path_sell}/sell_{curency}{textTime[j]}.csv', index=False)