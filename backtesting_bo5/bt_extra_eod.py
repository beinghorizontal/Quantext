import pandas as pd
import numpy as np
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objs as go


def btextra(df, trades):
    # df = df_daily.copy()
    # Merge based on nearest timestamp
    df['EntryTime'] = df.index
    df['ExitTime'] = df.index
    trades_entry = trades.copy()
    trades_entry.index = trades_entry['EntryTime']
    merged_entry = pd.merge_asof(trades_entry, df, right_on='EntryTime', left_index=True,
                                 direction='nearest')
    trades_exit = trades.copy()
    trades_exit.index = trades_exit['ExitTime']
    merged_exit = pd.merge_asof(trades_exit, df, right_on='ExitTime', left_index=True,
                                direction='nearest')

    """
    This code snippet is iterating over a list of trades and calculating the maximum notional 
    loss and maximum notional gain within a specific time range for each trade. 
    The `merged_df2` DataFrame is being sliced based on the entry and exit times of each trade.
     The `Low.min()` and `High.max()` functions are used to find the minimum and maximum values 
     of the `Low` and `High` columns within the specified time range. 
    The calculated values are then appended to the `maxloss` and `maxgain` lists respectively.
    Timedelta is used so it will exclude entry and exit bar. 
    """
    maxloss = []
    maxgain = []
    AR_list = []
    for i in range(len(trades)):
        # i =1
        max_notional_loss = df[merged_entry.shift(-1).index[i] : merged_exit.shift(1).index[i]].Low.min()
        max_notional_gain = df[merged_entry.shift(-1).index[i] : merged_exit.shift(-1).index[i]].High.max()
        ar_entry = df[merged_entry.shift(-1).index[i] : merged_exit.shift(-1).index[i]].iloc[0]['AR']

        maxloss.append(max_notional_loss)
        maxgain.append(max_notional_gain)
        AR_list.append(ar_entry)
    trades_extra = trades.copy()
    trades_extra['minlow'] = maxloss
    trades_extra['maxhigh'] = maxgain
    trades_extra['AR_Entry'] = AR_list
    # trades_extra['AR%']  = 100*((trades_extra['maxhigh'] - trades_extra['minlow']) / trades_extra['EntryPrice'])
    """
    Whenever a trade is completed within 30 minutes, the corresponding value in the 
    `maxloss` and `maxgain` columns is set to 0 by replacing NaN.
    """

    trades_extra['netmaxloss'] = np.where(trades_extra['Size'] > 0, trades_extra['minlow'] - trades_extra['EntryPrice'],
                                    trades_extra['EntryPrice'] - trades_extra['maxhigh'])
    trades_extra['netmaxgain'] = np.where(trades_extra['Size'] > 0, trades_extra['maxhigh'] - trades_extra['EntryPrice'],
                                    trades_extra['EntryPrice'] - trades_extra['minlow'])
    trades_extra['netmaxloss%'] = 100 * (trades_extra['netmaxloss'] / trades_extra['EntryPrice'])
    trades_extra['netmaxgain%'] = 100 * (trades_extra['netmaxgain'] / trades_extra['EntryPrice'])

    trades_extra['avg_pnl'] = trades_extra['PnL'] / abs(trades_extra['Size'])
    trades_extra['avg_gain'] = np.where(trades_extra['avg_pnl'] > 0, trades_extra['avg_pnl'], 0)
    trades_extra['avg_loss'] = np.where(trades_extra['avg_pnl'] < 0, trades_extra['avg_pnl'], 0)
    trades_extra['avg_gain%'] = 100 * (trades_extra['avg_gain'] / trades_extra['EntryPrice'])
    trades_extra['avg_loss%'] = 100 * (trades_extra['avg_loss'] / trades_extra['EntryPrice'])
    trades_extra['position_long'] = np.where(trades_extra['Size'] > 0, 1, 0)
    trades_extra['position_short'] = np.where(trades_extra['Size'] < 0, 1, 0)
    trades_extra['position_profit'] = np.where(trades_extra['PnL'] > 0, 1, 0)
    trades_extra['position_loss'] = np.where(trades_extra['PnL'] < 0, 1, 0)

    total_longs = trades_extra['position_long'].sum()
    total_shorts = trades_extra['position_short'].sum()
    total_wins = trades_extra['position_profit'].sum()
    total_losses = trades_extra['position_loss'].sum()

    avg_win = trades_extra['avg_gain'].sum() / total_wins
    avg_loss = trades_extra['avg_loss'].sum() / total_losses
    avg_winloss = abs(avg_win / avg_loss)

    trades_extra['ConsecutiveWinners'] = trades_extra.groupby((trades_extra['position_profit'] != trades_extra[
        'position_profit'].shift()).cumsum())['position_profit'].cumsum()

    trades_extra['ConsecutiveLosers'] = trades_extra.groupby((trades_extra['position_loss'] !=
                                                  trades_extra['position_loss'].shift()).cumsum(
    ))['position_loss'].cumsum()

    max_cons_winners = trades_extra['ConsecutiveWinners'].max()
    max_cons_losers = trades_extra['ConsecutiveLosers'].max()


    fill_values = {
        'maxhigh': 0,
        'minlow': 0,
        'netmaxgain': 0,
        'netmaxloss': 0,
        'netmaxgain%': 0,
        'netmaxloss%': 0
    }

    trades_extra.fillna(value=fill_values, inplace=True)

    # plot scatterplot with max notional loss and profit in percentage
    trades_extra['pnl_color'] = np.where(trades_extra['PnL'] > 0, 'limegreen', 'magenta')
    trades_extra['EntryTime'] = trades_extra['EntryTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    trades_extra['EntryTime_str'] = trades_extra['EntryTime'].astype(str)
    trades_extra['EntryTime_str'] = '<br>' + trades_extra['EntryTime_str']
    trades_extra['Size_str'] = trades_extra['Size'].astype(str)
    trades_extra['Size_str'] = '<br>Size: ' + trades_extra['Size_str']
    trades_extra['netmaxloss_abs%'] = abs(trades_extra['netmaxloss%']).round(2)
    trades_extra['netmaxgain_abs%'] = abs(trades_extra['netmaxgain%']).round(2)
    trades_extra['netmaxloss_str%'] = '<br>NotionalLoss: ' + trades_extra['netmaxloss_abs%'].astype(str) + '%'
    trades_extra['netmaxgain_str%'] = '<br>NotionalGain: ' + trades_extra['netmaxgain_abs%'].astype(str) + '%'
    trades_extra['avg_gain_str%'] = (trades_extra['avg_gain%'].round(2)).astype(str)
    trades_extra['avg_gain_str%'] = '<br>Avg_Gain: ' + trades_extra['avg_gain_str%'] + '%'
    trades_extra['avg_loss_str%'] = (trades_extra['avg_loss%'].round(2)).astype(str)
    trades_extra['avg_loss_str%'] = '<br>Avg_Loss: ' + trades_extra['avg_loss_str%'] + '%'

    avg_win_percentage = trades_extra['avg_gain%'].sum() / total_wins
    avg_loss_percentage = trades_extra['avg_loss%'].sum() / total_losses

    fig = make_subplots(rows=1, cols=1, shared_xaxes=False, row_heights=[1.0],
                        vertical_spacing=0.0003, horizontal_spacing=0.0003,
                        specs=[[{"secondary_y": False}]])

    fig.add_trace(go.Scatter(
        x=trades_extra['netmaxloss_abs%'],
        y=trades_extra['netmaxgain_abs%'],
        mode='markers',
        hovertext=trades_extra['EntryTime_str'] + trades_extra['Size_str'] + trades_extra['avg_gain_str%'] + trades_extra['avg_loss_str%'],
        marker=dict(color=trades_extra['pnl_color'], symbol='circle'),
        name='max notional loss and profit in percentage',
        showlegend=False
    ),col=1,row=1
    )
    #
    # fig.add_trace(go.Scatter(
    #     x=trades_extra['EntryTime'],
    #     y=trades_extra['AR_Entry'],
    #     mode='markers',
    #     hovertext=trades_extra['EntryTime_str'] + trades_extra['Size_str'] + trades_extra['avg_gain_str%'] + trades_extra['avg_loss_str%'],
    #     marker=dict(color=trades_extra['pnl_color'], symbol='circle'),
    #     showlegend=False
    # ),row=2,col=1
    # )

    fig.update_xaxes(title='Max Adverse Excursion%', showline=False, color='white', showgrid=False,
                     showticklabels=True,
                     rangeslider_visible=False,
                     tickangle=90,row=1,col=1)
    #
    # fig.update_xaxes(title='Date', showline=False, color='white', showgrid=False,
    #                  showticklabels=True,
    #                  rangeslider_visible=False,
    #                  tickangle=90,row=2,col=1)

    fig.update_yaxes(title='Max Favorable Excursion%', color='white', showgrid=False,
                     showticklabels=True, row=1, col=1)

    # fig.update_yaxes(title='AR_During_Entry', color='white', showgrid=False,
    #                  showticklabels=True, row=2, col=1)

    fig.update_layout(title='Max Adv. vs fav. Excursion', paper_bgcolor='black',
                      plot_bgcolor='black',
                      autosize=True, uirevision=True
                      )

    stats_new = (f"Total_longs: {total_longs}\nTotal_Shorts: {total_shorts}\nAvg_Win:"
                 f" {round(avg_win, 2)}\nAvg_Loss: {round(avg_loss, 2)}\nAvg_Win%: "
                 f"{round(avg_win_percentage,2)}\n"
                 f"Avg_Loss%: {round(avg_loss_percentage,2)}\n"
                 f"Avg_Win/Loss: {round(avg_winloss, 2)}\n"
                 f"Max_Consecutive_Winners: {max_cons_winners}\n"
                 f"Max_Consecutive_Losers: {max_cons_losers}\n")
    return stats_new, fig

def equity_curve_plot(df, df2):
    # df = ec.copy()
    # df2 = merged_df2.copy()
    df['DrawdownPct'] = 100 * df['DrawdownPct']
    df['rolling_mean'] = df['DrawdownPct'].rolling(10).mean()
    df['color'] = np.where(df['rolling_mean'] > df['rolling_mean'].shift(1), 'red', 'green')
    df['color'] = np.where(df['rolling_mean'] == df['rolling_mean'].shift(1), 'darkslategrey', df['color'])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Equity Curve vs Drawdown'), row_heights=[0.70, 0.30],
                        vertical_spacing=0.003,
                        horizontal_spacing=0.0003,
                        specs=[[{"secondary_y": True}], [{"secondary_y": False}]])


    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.Equity,
        mode='lines',
        name='EquityCurve',
        line=dict(color='cyan',
                  width=4,
                  # dash = 'dot'
                  ),
        hovertext=df['Equity'].round(2).astype(str)+'<br>'+df.index.astype(str),
        # hoverinfo='text',
        showlegend=False
    ),
        secondary_y=False, col=1, row=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df2.Close,
        mode='lines',
        name='Underlying',
        line=dict(color='white',
                  width=4,
                  dash = 'dot'
                  ),
        hovertext=df['Equity'].round(2).astype(str)+'<br>'+df.index.astype(str),
        # hoverinfo='text',
        showlegend=False
    ),
        secondary_y=True, col=1, row=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.DrawdownPct,
        mode='lines+markers',
        line=dict(color='gray', width=1),
        marker=dict(color=df['color'], size=4),
        name='DD',
        hovertext=df['DrawdownPct'].round(2).astype(str)+'<br>'+df.index.astype(str),
        showlegend=False
    ),
        secondary_y=False, col=1, row=2)

    fig.update_xaxes(showline=False, color='white', showgrid=False, showticklabels=False,
                     type='category', rangeslider_visible=False,
                     tickangle=90, zeroline=False, col=1, row=1)

    fig.update_xaxes(showline=False, color='white', showgrid=False, showticklabels=False,
                     type='category', rangeslider_visible=False,
                     tickangle=90, zeroline=False, col=1, row=2)

    fig.update_yaxes(title='EquityCurve', color='cyan', showgrid=False,
                     zeroline=False, showticklabels=True, row=1, col=1, tickformat=',d')

    fig.update_yaxes(title='Underlying', color='white', showgrid=False,
                     zeroline=False, showticklabels=True, row=1, col=1, tickformat=',d', secondary_y=True)

    fig.update_yaxes(title='DrawDown%', color='white', showgrid=False,
                     zeroline=False, showticklabels=True, row=2, col=1)

    fig.update_layout(title='EquityCurve vs DD', paper_bgcolor='black', plot_bgcolor='black',
                      autosize=True, uirevision=True
                      )
    # plot(fig)
    return fig

def backtest_plot(df, trades, indicator_plot = 'bar', symbol=""):
    # merged_list = [group[1] for group in df.groupby(df.index.date)]
    # df = merged_df2.copy()
    trades['color'] = np.where(trades.Size > 0, 'limegreen', 'magenta')
    trades['color'] = trades['color'].astype(str)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.70, 0.30],
                        vertical_spacing=0.003, horizontal_spacing=0.0003,
                        specs=[[{"secondary_y": False}], [{"secondary_y": False}]])

    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 showlegend=False,
                                 hovertext='VAH: '+ df.VAH.round(2).astype(str) + '<br>' + 'POC: ' + df.POC.astype(str) + '<br>' +
                                           'VAL: ' + df.VAL.astype(str),
                                 name=symbol, opacity=0.3), row=1, col=1)


    if indicator_plot == 'bar':
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['indicator'],
                name='DailyStrength',
                showlegend=False,
                hovertext=df.indicator.round(2).astype(str),
                marker=dict(
                    # color=df['color_power'],  # Using the column values for color
                    colorscale='Viridis',  # Specify the desired color scale
                )
            ),
            secondary_y=False, col=1, row=2)
    else:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['indicator'],
            mode='lines+markers',
            line=dict(color='gray', width=1),
            marker=dict(colorscale='Viridis', size=4),
            name='Indicator',
            hovertext=df.indicator.round(2).astype(str),
            showlegend=False
        ),
            secondary_y=False, col=1, row=2)




    # for df in merged_list:
    #     hovertext_vah = (f"VAH: {df.iloc[0]['VAH']}<br>"
    #                      )
    #
    #     fig.add_trace(go.Scatter(
    #         x=[df.iloc[0]['date'], df.iloc[-1]['date']],
    #         y=[df.iloc[0]['VAH'], df.iloc[0]['VAH']],
    #         mode='lines',
    #         line=dict(color='green',
    #                   width=1,
    #                   dash='dot'),
    #         hovertext=hovertext_vah,
    #         hoverinfo='text',
    #         showlegend=False
    #     ))
    #
    #     hovertext_val = (f"VAL: {df.iloc[0]['VAL']}<br>"
    #                      )
    #
    #     fig.add_trace(go.Scatter(
    #         x=[df.iloc[0]['date'], df.iloc[-1]['date']],
    #         y=[df.iloc[0]['VAL'], df.iloc[0]['VAL']],
    #         mode='lines',
    #         line=dict(color='red',
    #                   width=1,
    #                   dash='dot'),
    #         hovertext=hovertext_val,
    #         hoverinfo='text',
    #         showlegend=False
    #     ))

    for j in range(len(trades)):
        hovertext_trades = (f"EntryPrice: {round(trades.iloc[j]['EntryPrice'],2)}<br>"
                            f"Size: {trades.iloc[j]['Size']}<br>"
                            f"ExitPrice: {round(trades.iloc[j]['ExitPrice'],2)}<br>"
                            f"PnL: {round(trades.iloc[j]['PnL'],2)}")

        fig.add_trace(go.Scatter(
            x=[trades.iloc[j]['EntryTime'], trades.iloc[j]['ExitTime']],
            y=[trades.iloc[j]['EntryPrice'], trades.iloc[j]['ExitPrice']],
            mode='lines',
            line=dict(color=trades['color'][j],
                      width=3,
                      # dash = 'dot'
                      ),
            hovertext=hovertext_trades,
            hoverinfo='text',
            showlegend=False
        ))

    fig.update_xaxes(showline=False, color='white', showgrid=False, showticklabels=False,
                     type='category', rangeslider_visible=False,
                     tickangle=90, zeroline=False, col=1, row=1)

    fig.update_xaxes(showline=False, color='white', showgrid=False, showticklabels=False,
                     type='category', rangeslider_visible=False,
                     tickangle=90, zeroline=False, col=1, row=2)

    fig.update_yaxes(color='white', showgrid=False,
                     zeroline=False, showticklabels=True, row=1, col=1, tickformat=',d')

    fig.update_yaxes(color='white', showgrid=False,
                     zeroline=False, showticklabels=True, row=2, col=1, tickformat=',d')

    fig.update_layout(paper_bgcolor='black', plot_bgcolor='black',
                      autosize=True, uirevision=True
                      )
    # plot(fig)
    return fig
