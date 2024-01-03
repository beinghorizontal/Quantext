import datetime
import pandas as pd
import requests
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from io import BytesIO
from zipfile import ZipFile
import numpy as np

path = 'd:/demos/storage/hl'
dayback = 1

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
newhs = []
newls = []

text1 = ''
# for i in range(0,90):
try:
    dt_1 = datetime.date.today() - datetime.timedelta(dayback)  # 1 for yday and 0 for today
    day_nse = dt_1.strftime("%d%m%y")

    # raw_url = "https://nsearchives.nseindia.com/archives/equities/mkt/MA221223.csv"
    # raw_url_hl= "https://nsearchives.nseindia.com/archives/equities/bhavcopy/pr/PR291223.zip"

    url = "https://nsearchives.nseindia.com/archives/equities/mkt/MA" + day_nse + ".csv"
    urlHL = "https://nsearchives.nseindia.com/archives/equities/bhavcopy/pr/PR" + day_nse + ".zip"

    headers = {'Connection': 'keep-alive',
               'authority': 'www.nseindia.com',
               'path': '/api/marketStatus',
               'Origin': 'https://www1.nseindia.com',
               'Referer': 'https://www1.nseindia.com/products/content/equities/equities/archieve_eq.htm',
               'Sec-Fetch-Mode': 'cors',
               'Sec-Fetch-Site': 'same-origin',
               'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

    req = requests.get(url, headers=headers)
    response = requests.get(urlHL, headers=headers)  # to fetch 53 week H/L data

    if req.status_code == 200 and response.status_code==200:  # If the response for both url is successful
        # Since H/L url is in zip format we need to unzip it first, for that we use ZipFile library
        with ZipFile(BytesIO(response.content)) as zip_file:
            # Extract all files in the zip to a temporary directory
            zip_file.extractall(path)

        csv_path = path + '/HL' + day_nse + '.csv'
        dfHL = pd.read_csv(csv_path)
        dfHL['numeric'] = np.where(dfHL['NEW_STATUS'] == 'H', 1, 0)
        newh = dfHL['numeric'].sum()
        newl = len(dfHL) - newh

        list_date = req.text.split('\n')
        df = pd.DataFrame(list_date)
        text1 = df[0:6][0].str.cat()

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
        newhs.append(newh)
        newls.append(newl)
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
                    'trades_list': trades_list, 'new_high':newhs, 'new_low':newls})

df_hist = pd.read_csv('d:/demos/nifty_index_data.csv')
dfmerge = pd.concat([df_hist, df1])
# dfmerge.to_csv('d:/demos/nifty_index_data.csv', index=False, mode='w')

dfmerge['ad_ratio'] = dfmerge['adv_list'] / dfmerge['dec_list']
dfmerge['hl_ratio'] = dfmerge['new_high'] / dfmerge['new_low']

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,start_cell="top-left",
                       subplot_titles=('Nifty', 'Volume'),
                       row_heights=[0.55, 0.23], vertical_spacing=0.003,
                       horizontal_spacing=0.03,
                       specs=[[{"secondary_y": True}],
                              [{"secondary_y": True}]])

# fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Candlestick(x=dfmerge['date'], open=dfmerge['open_list'], high=dfmerge['high_list'],
                             low=dfmerge['low_list'],
                             close=dfmerge['close_list'],name='Nifty'),col=1,row=1)  # candlestick
fig.add_trace(go.Scatter(x=dfmerge.date, y=dfmerge['ad_ratio'], mode='lines', name = 'Advance-Decline Ratio'),
              secondary_y=True,col=1,row=1)
fig.update_yaxes( showgrid=False, color='white')
fig.add_trace(go.Scatter(x=dfmerge.date, y=dfmerge['hl_ratio'], mode='lines', name = '52wk High/Low ratio'),
              secondary_y=True,col=1,row=2)

fig.add_trace(go.Bar(x=dfmerge.date, y=dfmerge['qty_list'], name = 'Traded Volume'), secondary_y=False,col=1,row=2)

fig.update_layout(paper_bgcolor='black',
                  title = '<a href="https://www.youtube.com/@quantext/featured">Youtube @quantext</a>',
                  plot_bgcolor='black', xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False), yaxis2=dict(range=[0, 10], showgrid=False))
fig.layout.xaxis.type = 'category'
fig.update_layout(yaxis_tickformat='d')  # d for showing full digits else it will show 19000 as 19k


fig["layout"]["xaxis"]["rangeslider"]["visible"] = False
fig.update_xaxes(showline=False, color='white', showgrid=False, type='category',
                            tickangle=90, col=1,zeroline=False, row=2)

def add_br_every_n_chars(text, n):
    result = ''
    for i, char in enumerate(text, start=1):
        result += char
        if i % n == 0:
            result += '<br>'
    return result

# Example usage

text = add_br_every_n_chars(text1, 31)

fig.update_layout(
    annotations=[
        dict(
            x=1.07,  # X-coordinate (1.0 is the right edge of the plot)
            y=0.3,   # Y-coordinate (0.5 is the middle of the plot)
            xref='paper',  # Use paper coordinates for x
            yref='paper',  # Use paper coordinates for y
            text=text,
            showarrow=False,
            font=dict(
                family="Arial",
                size=12,
                color="white"
            ),
        )
    ]
)

plot(fig, auto_open=True)
# fig.show()  # Uncomment this if you want to run inside Pycharm or ipython
