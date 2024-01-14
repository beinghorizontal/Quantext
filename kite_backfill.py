import requests
import pandas as pd
import datetime
from def_instrumentid import getinstrumentid
from kite_auto import kitelogin

kitelogin()
dayback = 10
fromday = (datetime.date.today() - datetime.timedelta(dayback)).strftime('%Y-%m-%d')
curday = (datetime.date.today() - datetime.timedelta(0)).strftime('%Y-%m-%d')
root = "Y:/FolderName/"

file_names = ['kite_userid.txt', 'kite_pass.txt', 'kite_totp.txt']
credentials = []
for file in file_names:
    with open(f'{root}{file}', 'r') as f:
        content = f.read()
        credentials.append(content)

userid = credentials[0]
instrument_id = getinstrumentid(symbol_name="NIFTY")[0]  # Check data/instruments.csv to change the name

url = f"https://kite.zerodha.com/oms/instruments/historical/{instrument_id}/minute?user_id={userid}&oi=1&from={fromday}&to={curday}"

with open('D:/File/Path/token.txt', 'r') as f:
    tokens = f.read()
token = tokens
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
           'Authorization':token}
response = requests.get(url, headers=headers)
print(response.status_code)
file=response.json()
data=file['data']
data=data['candles']
df=pd.DataFrame(data)
df.columns=['dt','o','h','l','c','v','oi']
df2=df.dt.str.split('T',expand=True)
df3=df[['o','h','l','c','v','oi']]
mdf=pd.concat([df2,df3],axis=1)
mdf.columns=['Date','Time','Open','High','Low','Close','Volume','OpenI']
mdf['Time'] = mdf['Time'].astype(str).str[:-5]
mdf.to_csv('d:/filePath/niftyf.csv', index=False)
