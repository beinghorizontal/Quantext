from datetime import timedelta

import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from plotly.offline import plot

from tpo_helper2 import (
    get_context,
    get_dayrank,
    get_ibrank,
    get_mean,
    get_rf,
    get_ticksize,
)

# Download intraday data for Nifty

nifty_ticker = "^NSEI"
tpo_spacing = 2

# Note: yfinance allows you to specify the interval. Common intervals are '1m', '5m', '15m', '30m', '60m'
data = yf.download(tickers=nifty_ticker, period="7d", interval="1m", ignore_tz=True)
# To get the TPO chart from local data change the file path to your csv file"
# filePath = 'd:/anaconda/Scripts/niftyf.csv'
# data = pd.read_csv(filePath, header=0)
# data['DateTime'] = pd.to_datetime(data['DateTime'], format='%Y-%m-%d %H:%M:%S')
# data = data.set_index('DateTime', drop=True, inplace=False)
datacolumns = ["Open", "High", "Low", "Close", "Volume"]
data.columns = datacolumns
# data.to_csv('temp.csv')
data.tail()
# Remove all rows where the datetime index starts from date 2025-11-04
# data = data[data.index.date < np.datetime64("2025-11-12").astype('M8[D]')]
# manual parameters
freq = 30
avglen = 7  # num days mean to get values
days_to_display = 7  # Number of last n days you want on the screen to display
mode = "tpo"  # for volume --> 'vol', for TPO --> 'tpo'. TPO is recommended for Indian markets.
# Volume is for global markets since volume in Indian markets is spikey and not reliable
# get tick size based on most recent data. No need to change parameters for global markets"
ticksz = get_ticksize(data, freq=freq)
ticksz = ticksz + tpo_spacing
# symbol = symbol_name
mean_val = get_mean(data, avglen=avglen, freq=freq)
trading_hr = mean_val["session_hr"]

# !!! get rotational factor again for 30 min resampled data
data = get_rf(data.copy())
# !!! resample to desire time frequency. For TPO charts 30 min is optimal
dfresample = (
    data.copy()
)  # create seperate resampled data frame and preserve old 1 min file
dfresample["datetime"] = dfresample.index
dfresample = dfresample.resample(str(freq) + "min").agg(
    {
        "datetime": "last",
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
        "rf": "sum",
    }
)
dfresample = dfresample.dropna()

# slice df based on days_to_display parameter
dt1 = dfresample.index[-1]
sday1 = dt1 - timedelta(days_to_display)
dfresample = dfresample[(dfresample.index.date > sday1.date())]

# !!! split the dataframe with new date
DFList = [group[1] for group in dfresample.groupby(dfresample.index.date)]
# !!! for context based bubbles at the top with text hovers
dfcontext = get_context(
    dfresample, freq=freq, ticksize=ticksz, style=mode, session_hr=trading_hr
)
dfmp_list = dfcontext[0]
df_distribution = dfcontext[1]
df_ranking = get_dayrank(df_distribution.copy(), mean_val)
ranking = df_ranking[0]

power1 = ranking.power1  # Non-normalised day's strength
power = ranking.power  # Normalised day's strength for dynamic shape size for markers
breakdown = df_ranking[1]
dh_list = ranking.highd
dl_list = ranking.lowd
# !!! get context based on IB It is predictive value caculated by using various IB stats and previous day's value area
# IB is 1st 1 hour of the session. Not useful for scrips with global 24 x 7 session
context_ibdf = get_ibrank(mean_val, ranking)
ibpower1 = context_ibdf[0].ibpower1  # Non-normalised IB strength
ibpower = context_ibdf[
    0
].IB_power  # Normalised IB strength for dynamic shape size for markers at bottom
ibbreakdown = context_ibdf[1]
ib_high_list = context_ibdf[0].ibh
ib_low_list = context_ibdf[0].ibl

# Empty the file before writing
with open("e:/envs/market_data.txt", "w") as f:
    pass
symbol = "NiftyF"

fig = go.Figure()

fig = go.Figure(
    data=[
        go.Candlestick(
            x=dfresample["datetime"],
            open=dfresample["Open"],
            high=dfresample["High"],
            low=dfresample["Low"],
            close=dfresample["Close"],
            showlegend=False,
            name=symbol,
            opacity=0.8,
        )
    ]
)


# !!! get TPO for each day
for i in range(len(dfmp_list)):  # test the loop with i=0
    df1 = DFList[i].copy()
    df_mp = dfmp_list[i]
    irank = ranking.iloc[i]
    irank_prev = ranking.iloc[i - 1] if i > 0 else None
    # df_mp['i_date'] = df1['datetime'][0]
    df_mp["i_date"] = irank.date
    # # @todo: background color for text
    df_mp["color"] = np.where(
        np.logical_and(df_mp["close"] > irank.vallist, df_mp["close"] < irank.vahlist),
        "green",
        "white",
    )

    df_mp = df_mp.set_index("i_date", inplace=False)

    fig.add_trace(
        go.Scatter(
            x=df_mp.index,
            y=df_mp.close,
            mode="text",
            name=str(df_mp.index[0]),
            text=df_mp.alphabets,
            showlegend=False,
            textposition="top right",
            textfont=dict(family="verdana", size=18, color=df_mp.color),
        )
    )
    # if power1[i] < 0:
    #     my_rgb = "rgba({power}, 3, 252, 0.5)".format(power=abs(165))
    # else:
    #     my_rgb = "rgba(23, {power}, 3, 0.5)".format(power=abs(252))
    if i > 0:
        lvnlist_str = list(map(str, irank.lvnlist))

        fig.add_trace(
            go.Scatter(
                x=df_mp.index,
                y=[
                    df1["High"].max() * 1.0005
                ],  # position of the text box on y axis with slight offset
                mode="text",
                text=[
                    "<br />Date: {}<br />VAH:  {}<br /> POC:  {}<br /> VAL:  {}<br />yVAH:  {}<br />yVAL:  {}<br />yPOC:  {}<br />Open:  {}<br />High:  {}<br /> Low:  {}<br />Close:  {}".format(
                        irank.date,
                        int(irank.vahlist),
                        int(irank.poclist),
                        int(irank.vallist),
                        int(irank_prev.vahlist),
                        int(irank_prev.vallist),
                        int(irank_prev.poclist),
                        int(df1.iloc[0]["Open"]),
                        int(irank.highd),
                        int(irank.lowd),
                        int(irank.close),
                    )
                ],
                textposition="top right",
                textfont=dict(size=16, color="white"),
                showlegend=False,
            )
        )

        # Save market data to text file
        market_data = {
            "Date": irank.date,
            "VAH": int(irank.vahlist),
            "POC": int(irank.poclist),
            "VAL": int(irank.vallist),
            "Yesterday's VAH": int(irank_prev.vahlist),
            "Yesterday's VAL": int(irank_prev.vallist),
            "Yesterday's POC": int(irank_prev.poclist),
            "Open": int(df1.iloc[0]["Open"]),
            "High": int(irank.highd),
            "Low": int(irank.lowd),
            "Close": int(irank.close),
            "Daily_Range": int(irank.ranged),
            "Daily_Rotation_factor": int(irank.rfd),
            "IBR_High": int(irank.ibh),
            "IBR_Low": int(irank.ibl),
            "LVNs": lvnlist_str,
        }
        # Write to file
        with open("e:/envs/market_data.txt", "a") as f:
            for key, value in market_data.items():
                (f.write(f"{key}: {value}\n"),)
        with open("e:/envs/market_data.txt", "a") as f:
            f.write("---------------------\n")


ltp = df1.iloc[-1]["Close"]
ltp = int(ltp)
if ltp >= irank.poclist:
    ltp_color = "lightgreen"
else:
    ltp_color = "magenta"

fig.add_trace(
    go.Scatter(
        x=[df1.iloc[-1]["datetime"]],
        y=[df1.iloc[-1]["Close"]],
        mode="text",
        name="last traded price",
        text=["last " + str(ltp)],
        textposition="bottom right",
        textfont=dict(size=16, color=ltp_color),
        showlegend=False,
    )
)

fig.layout.xaxis.color = "white"
fig.layout.yaxis.color = "white"
fig.layout.autosize = True
# fig["layout"]["height"] = 800
# fig.layout.hovermode = 'x'
# fig.layout.plot_bgcolor = '#44494C'
# fig.layout.paper_bgcolor = '#44494C'
# Add extra space to the right of the plot


fig.update_yaxes(
    title_text="Nifty",
    tickformat="d",
    title_font=dict(size=18, color="white"),
    tickfont=dict(size=12, color="white"),
    showgrid=False,
    # Add 0.5% extra space at the top for text visibility
    range=[dfresample["Low"].min(), dfresample["High"].max() * 1.005],
)

fig.update_xaxes(
    showgrid=False,
    zeroline=False,
    rangeslider_visible=False,
    showticklabels=False,
    color="white",
    type="category",
    tickangle=90,
    dtick=30,  # Control spacing between ticks
)

fig.update_layout(
    paper_bgcolor="black",
    plot_bgcolor="black",
    autosize=True,
    uirevision=True,
    # margin=dict(r=480)
)
# fig.update_layout(
#     margin=dict(r=120)  # Increase right margin (default is 80)
# )

plot(fig, auto_open=True)
fig.show()
fig.write_image(
    "e:/envs/nifty_tpo_plot.png", width=1920, height=1920
)  # to write the image in png format install kaleido package
