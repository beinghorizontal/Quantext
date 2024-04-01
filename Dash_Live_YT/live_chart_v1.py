"""
Import libraries
"""
# pip install dash
import dash
from dash.dependencies import Input, Output, State
from dash import html, dcc
import pandas as pd
from plotly.subplots import make_subplots
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

"""

token = 0
replay = False
replay_speed_bars = 10
#%%
dfhist = get_data(symbol_name="NIFTY", history=15, current_day=0, token=token)
dfhist['DateTime'] = pd.to_datetime(dfhist['DateTime'], format='%Y-%m-%dT%H:%M:%S%z')
dfhist = dfhist.set_index('DateTime', drop=True, inplace=False)
# %%
app = dash.Dash(__name__)

app.layout = html.Div(
    html.Div([
        dcc.Location(id='url', refresh=False),
        dcc.Link('For questions, ping me on YouTube', href='https://youtube.com/@quantext'),
        html.Br(),
        dcc.Link('FAQ and python source code', href='http://www.github.com/beinghorizontal/Quantext/dash'),
        html.H4('@Quantext'),
        dcc.Graph(id='Quantext'),
        dcc.Interval(
            id='interval-component',
            interval=5000,  # in milliseconds
            n_intervals=0
        )
    ])
)

#%%
"""
Anything you put below the function update_graph, it will get looped to fetch new data, so only use necessary functions
"""
@app.callback(Output(component_id='Quantext', component_property='figure'),
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
                                  [{"secondary_y": False}]])


    if replay == False:

        df = get_data(symbol_name="NIFTY", history=0, current_day=0, token=0)
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%dT%H:%M:%S%z')
        df = df.set_index('DateTime', drop=True, inplace=False)
        df_merge = pd.concat([dfhist, df], axis=0)
        # remove duplicates of index and keep the last one for dataframe df
        df_merge = df_merge.drop_duplicates()
    else:
        last_date = dfhist.index[-1].date()
        df = dfhist[dfhist.index.date == last_date]
        dfhist2 = dfhist[dfhist.index.date != last_date]
        # n= 0
        n +=replay_speed_bars
        df_merge = pd.concat([dfhist2, df.head(n)], axis=0)

    subfig.add_trace(go.Candlestick(x=df_merge.index, open=df_merge['Open'],
                                   high=df_merge['High'], low=df_merge['Low'],
                                   close=df_merge['Close'], name=symbol,
                                    showlegend=False,
                                   increasing_line_color='#17B897', decreasing_line_color='#F44336',
                                   line_width=1),row=1,col=1,
                                    secondary_y=False)

    # C chnage y axis col and row
    subfig.add_trace(
        go.Bar(
            x=df_merge.index,
            y=df_merge['Volume'],
            name='span style="color:white">delta Diff</span>',
            showlegend=False,
            marker=dict(
                color=df_merge['Volume'],  # Using the column values for color
                colorscale='Viridis',  # Specify the desired color scale
            )
        ),
    secondary_y=False,col=1,row=2)

    subfig.add_trace(
        go.Bar(
            x=df_merge.index,
            y=df_merge['OpenI'],
            name='span style="color:white">delta Diff</span>',
            showlegend=False,
            marker=dict(
                color=df_merge['OpenI'],  # Using the column values for color
                colorscale='Viridis',  # Specify the desired color scale
            )
        ),
    secondary_y=False,col=1,row=3)

    last_closing_price = df_merge.iloc[-1]['Close']
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
    # plot(subfig)

#%%

if __name__ == '__main__':
    app.run_server(port=8026, host='127.0.0.1', debug=True)
    # app.run_server(port=8052, host='0.0.0.0', debug=True)

