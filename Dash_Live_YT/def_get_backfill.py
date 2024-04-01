#%%
import requests
import pandas as pd
import datetime
from def_instrumentid import getinstrumentid
from kite_auto import kitelogin
#%%
#%%


def get_data(symbol_name="NIFTY", history=10, current_day=0, token=1):
    # %%
    if token == 1:
        kitelogin()
#%%
    dayback = history
    fromday = (datetime.date.today() - datetime.timedelta(dayback)).strftime('%Y-%m-%d')
    curday = (datetime.date.today() - datetime.timedelta(current_day)).strftime('%Y-%m-%d')
    root = "Z:/kite/"
#%%
    file_names = ['kite_userid.txt', 'kite_pass.txt', 'kite_totp.txt']
    credentials = []
    for file in file_names:
        with open(f'{root}{file}', 'r') as f:
            content = f.read()
            credentials.append(content)

    userid = credentials[0]
#%%
    instrument_id = getinstrumentid(symbol_name=symbol_name)[0] # Check data/instruments.csv to change the name

    url = f"https://kite.zerodha.com/oms/instruments/historical/{instrument_id}/minute?user_id={userid}&oi=1&from={fromday}&to={curday}"

    with open('D:/anaconda/Scripts/token.txt', 'r') as f:
        tokens = f.read()
    token = tokens
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
               'Authorization':token}
    response = requests.get(url, headers=headers)
    print(response.status_code)

#%%
    file=response.json()
    data=file['data']
    data=data['candles']
    df=pd.DataFrame(data)
    df.columns=['DateTime','Open','High','Low','Close','Volume','OpenI']
    # df2=df.dt.str.split('T',expand=True)
    # df3=df[['o','h','l','c','v','oi']]
    # mdf=pd.concat([df2,df3],axis=1)
    # mdf.columns=['Date','Time','Open','High','Low','Close','Volume','OpenI']
    # mdf['Time'] = mdf['Time'].astype(str).str[:-5]
    return df
# mdf.to_csv('d:/filePath/niftyf.csv', index=False)

