import ast
import datetime
import re
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests

dayback = 0  # To download historical data from current date after market hours put dayback = -1
date_list = []
symbol_list = []
adv_list = []
dec_list = []
open_list = []
high_list = []
low_list = []
close_list = []
qty_list = []
trades_list = []

for i in range(0, 90):  # You can increase this period if you want, replace 90 with higher number, but I haven't tested
    try:
        dayback += 1

        dt_1 = datetime.date.today() - datetime.timedelta(dayback)  # 1 for y'day and 0 for today
        day_nse = dt_1.strftime("%d%m%y")
        # raw_url = "https://nsearchives.nseindia.com/archives/equities/mkt/MA221223.csv"

        url = "https://nsearchives.nseindia.com/archives/equities/mkt/MA" + day_nse + ".csv"
        headers = {'Connection': 'keep-alive',
                   'authority': 'www.nseindia.com',
                   'path': '/api/marketStatus',
                   'Origin': 'https://www1.nseindia.com',
                   'Referer': 'https://www1.nseindia.com/products/content/equities/equities/archieve_eq.htm',
                   'Sec-Fetch-Mode': 'cors',
                   'Sec-Fetch-Site': 'same-origin',
                   'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

        req = requests.get(url, headers=headers)
        if req.status_code == 200:

            list_date = req.text.split('\n')
            df = pd.DataFrame(list_date)
            nifty_val = df.iloc[9, 0]

            nifty_list = nifty_val.split(',')

            date_obj = pd.to_datetime(dt_1, format="%d%m-%Y")
            adv_str = df.iloc[81, 0]
            dec_str = df.iloc[82, 0]
            adv_r = adv_str.split(",")[2]
            dec_r = dec_str.split(",")[2]
            adv = float(adv_r)
            dec = float(dec_r)

            sym = nifty_list[1]
            o = float(nifty_list[3])
            h = float(nifty_list[4])
            l = float(nifty_list[5])
            c = float(nifty_list[6])
            qty = float(df.iloc[4, 0].split(",")[2])
            trades = float(df.iloc[5, 0].split(",")[2])

            # append to lists
            date_list.append(date_obj)
            symbol_list.append(sym)
            open_list.append(o)
            high_list.append(h)
            low_list.append(l)
            close_list.append(c)
            qty_list.append(qty)
            trades_list.append(trades)
            adv_list.append(int(adv))
            dec_list.append(int(dec))

        else:
            print('connection error')
    except Exception as e:
        print(e)

df1 = pd.DataFrame({'date': date_list, 'symbol_list': symbol_list,
                    'adv_list': adv_list,
                    'dec_list': dec_list, 'open_list': open_list,
                    'high_list': high_list, 'low_list': low_list,
                    'close_list': close_list, 'qty_list': qty_list,
                    'trades_list': trades_list})

# df1 = df1.set_index('date')
df2 = df1.sort_index(ascending=False)
df2.to_csv('d:/demos/nifty_index_data.csv', index=False)

# df2 = df2.reset_index()
# df2.close_list.plot()
