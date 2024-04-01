# %%

"""
Import libraries
"""
import dash
from dash.dependencies import Input, Output, State
from dash import html, dcc
import pandas as pd
from plotly.subplots import make_subplots
import numpy as np
import datetime
import math
import time
import threading
from get_of_history import get_history
import plotly.graph_objs as go
from plotly.offline import plot
from def_get_backfill import get_data

# %%
"""
Inputs:
Token: Get historical data, use token = 0 if you are running the code within 24 hrs else use 1 so it will fetch new tokens
for your id, it expects that you have already saved Zerodha user id, password, and totp in a local or remotely mapped disk.
This takes 10 extra seconds, so do not run it too frequently.

replay: True or False. If true, it will replay the chart with the latest data and then update to simulate live data.  

replay_speed_bars: number of bars to replay at a time.

freq: frequency of the chart, default is 15min
"""

token = 1
replay = False
replay_speed_bars = 10
freq = 15
#%%
"""
Get rotation factor
"""
def get_rf(df):
    df = dfhist.copy()
    df['cup'] = np.where(df['Close'] >= df['Close'].shift(), 1, -1)
    df['hup'] = np.where(df['High'] >= df['High'].shift(), 1, -1)
    df['lup'] = np.where(df['Low'] >= df['Low'].shift(), 1, -1)

    df['Rotation_Factor'] = df['cup'] + df['hup'] + df['lup']
    df = df.drop(['cup', 'lup', 'hup'], axis=1)
    return df

dfhist = get_data(symbol_name="NIFTY", history=15, current_day=0, token=token)
dfhist['DateTime'] = pd.to_datetime(dfhist['DateTime'], format='%Y-%m-%dT%H:%M:%S%z')
dfhist = dfhist.set_index('DateTime', drop=True, inplace=False)
# create new column with volume delta
dfhist['Volume_Delta'] = dfhist['Volume'].diff()
dfhist  = dfhist.fillna(0)
dfhist = get_rf(dfhist)

#%%
"""
Resample to higher time frame, default is 15 min
"""
# resample open high low close from 1 minute to 15 minute for dfhist
dfhist_resample = dfhist.resample(str(freq)+'min').agg({'Open': 'first', 'High': 'max',
                                               'Low': 'min', 'Close': 'last', 'Volume_Delta': 'sum', 'Volume': 'sum', 'Rotation_Factor': 'sum'})

dfhist_resample = dfhist_resample.dropna()

# %%
app = dash.Dash(__name__)

app.layout = html.Div(
    html.Div([
        dcc.Location(id='url', refresh=False),
        dcc.Link('For questions, ping me on Twitter', href='https://twitter.com/beinghorizontal'),
        html.Br(),
        dcc.Link('FAQ and python source code', href='https://github.com/beinghorizontal/Quantext/new/main/Dash_Live_YT'),
        html.H4('@beinghorizontal'),
        dcc.Graph(id='beinghorizontal'),
        dcc.Interval(
            id='interval-component',
            interval=10000,  # in milliseconds
            n_intervals=0
        )
    ])
)

#%%
"""
Anything you put below the function update_graph, it will get looped to fetch new data, so only use necessary functions
"""
@app.callback(Output(component_id='beinghorizontal', component_property='figure'),
              [Input('interval-component', 'n_intervals')])

def update_graph(n):
    #C This is the loop

    # df_live_list = []
    symbol = 'Nifty'
    subfig = make_subplots(rows=3, cols=1, start_cell="top-left",
                           subplot_titles=(symbol),
                           row_heights=[0.50, 0.25,0.25],
                           vertical_spacing=0.003,
                           horizontal_spacing=0.03,
                           specs=[[{"secondary_y": False}],
                                  [{"secondary_y": False}],
                                  [{"secondary_y": True}]])


    if replay == False:

        df = get_data(symbol_name="NIFTY", history=0, current_day=0, token=0)
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%dT%H:%M:%S%z')
        df = df.set_index('DateTime', drop=True, inplace=False)
        df_merge = pd.concat([dfhist, df], axis=0)
        # remove duplicates of index and keep the last one for dataframe df
        df_merge = df_merge.drop_duplicates()
        dfmerge_resample = df_merge.resample(str(freq) + 'min').agg({'Open': 'first', 'High': 'max',
                                                                  'Low': 'min', 'Close': 'last', 'Volume_Delta': 'sum',
                                                                  'Volume': 'sum', 'Rotation_Factor': 'sum'})

        # Sum the remaining columns within each 1-minute interval
        dfmerge_resample = dfmerge_resample.dropna()
        dfmerge_resample['RF_cumsum'] = dfmerge_resample['Rotation_Factor'].cumsum()

    else:
        last_date = dfhist.index[-1].date()
        df = dfhist[dfhist.index.date == last_date]
        dfhist2 = dfhist[dfhist.index.date != last_date]
        # n= 0
        n +=replay_speed_bars
        df_merge = pd.concat([dfhist2, df.head(n)], axis=0)
        dfmerge_resample = df_merge.resample(str(freq) + 'min').agg({'Open': 'first', 'High': 'max',
                                                                  'Low': 'min', 'Close': 'last', 'Volume_Delta': 'sum',
                                                                  'Volume': 'sum', 'Rotation_Factor': 'sum'})

        # Sum the remaining columns within each 1-minute interval
        dfmerge_resample = dfmerge_resample.dropna()
        dfmerge_resample['RF_cumsum'] = dfmerge_resample['Rotation_Factor'].cumsum()

    subfig.add_trace(go.Candlestick(x=dfmerge_resample.index, open=dfmerge_resample['Open'],
                                   high=dfmerge_resample['High'], low=dfmerge_resample['Low'],
                                   close=dfmerge_resample['Close'], name=symbol,
                                    showlegend=False,
                                   increasing_line_color='#17B897', decreasing_line_color='#F44336',
                                   line_width=1),row=1,col=1,
                                    secondary_y=False)

    # C chnage y axis col and row
    subfig.add_trace(
        go.Bar(
            x=dfmerge_resample.index,
            y=dfmerge_resample['Volume_Delta'],
            name='span style="color:white">delta Diff</span>',
            showlegend=False,
            marker=dict(
                color=dfmerge_resample['Volume'],  # Using the column values for color
                colorscale='Viridis',  # Specify the desired color scale
            )
        ),
    secondary_y=False,col=1,row=2)
    # C Don't forget to change col, rows and sec y true or false
    subfig.add_trace(
        go.Scatter(
            x=dfmerge_resample.index,
            y=dfmerge_resample['RF_cumsum'],
            mode='lines+markers',
            name='<span style="color:yellow">FII</span>',
            marker=dict(color=dfmerge_resample['RF_cumsum'], colorscale='Viridis',
                        line_color='yellow'),
        ),
    secondary_y=False,col=1,row=3)


    last_closing_price = dfmerge_resample.iloc[-1]['Close']
    subfig.add_hline(y=last_closing_price, col=1, row=1, secondary_y=False)
    subfig.add_vline(x=df.index[0], col=1, row=1, secondary_y=False)

    subfig.update_yaxes(showgrid=False, zeroline=False, showticklabels=True,tickformat='d',
                        color='white',
                     # showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid',
                        col= 1, row=1,  secondary_y=False)

    subfig.update_xaxes(showgrid=False, zeroline=False, rangeslider_visible=False, showticklabels=False,color='black',
                        type='category',
                        tickangle=90, col=1, row=1)

    subfig.update_xaxes(showgrid=False, zeroline=False, rangeslider_visible=False, showticklabels=False,color='black',
                        type='category',
                        tickangle=90, col=1, row=2)

    subfig.update_xaxes(showgrid=False, zeroline=False, rangeslider_visible=False, showticklabels=False,color='black',
                        type='category',
                        tickangle=90, col=1, row=3)


    subfig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False,tickformat='d',
                        color='black',
                     # showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid',
                        col= 1, row=2,  secondary_y=False)

    subfig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False,tickformat='d',
                        color='black',
                     # showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid',
                        col= 1, row=3,  secondary_y=False)

    subfig.update_traces(xaxis='x1',  col=1, row=2)
    subfig.update_traces(xaxis='x1',  col=1, row=3)


    subfig.update_layout(paper_bgcolor='black', plot_bgcolor='black',
                         height=1080, width=1920,autosize=True,uirevision=True,
                        # legend=dict(y=1, x=0),
                        # hoverdistance=0,
                        # font=dict(color='white')
                        # margin=dict(b=20, t=0, l=0, r=40),
                         )
    return subfig
    plot(subfig)

#%%

if __name__ == '__main__':
    app.run_server(port=8028, host='127.0.0.1', debug=True)
    # app.run_server(port=8052, host='0.0.0.0', debug=True)  # if you use host - 0.0.0.0 then you can watch live sreaming chart on computer of mobile that is within same local network

