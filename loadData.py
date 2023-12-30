import pandas as pd
import numpy as np
import MetaTrader5 as mt
from datetime import datetime, timedelta
import pytz
import os
from function_indicator import GetAll_indicator

# Set the time zone to GMT+2
desired_timezone = pytz.timezone('Europe/Istanbul')

# Get the current time in the desired time zone
current_time = datetime.now(desired_timezone) + timedelta(days=1)  # Setting it to a future date

# Initialize MetaTrader5
print(mt.initialize())

login = 510131205
password = '!RXaRX74Xz'
server = 'FxPro-MT5'

mt.login(login, password, server)

EURUSD = 'EURUSD'
USDJPY = 'USDJPY'
GBPUSD = 'GBPUSD'
AUDUSD = 'AUDUSD'
USDCHF = 'USDCHF'
USDCAD = 'USDCAD'

Curency = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCHF', 'USDCAD']

interval_5m = mt.TIMEFRAME_M5
interval_15m = mt.TIMEFRAME_M15
interval_30m = mt.TIMEFRAME_M30
interval_1H = mt.TIMEFRAME_H1
interval_2H = mt.TIMEFRAME_H2
interval_4H = mt.TIMEFRAME_H4

time_frame = [interval_5m, interval_15m, interval_30m, interval_1H, interval_2H, interval_4H]
textTime = ['5m', '15m', '30m', '1h', '2h', '4h']

# Set a large number for historical data
no_of_row = 99999