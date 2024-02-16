import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from screeninfo import get_monitors

from def_get_marketParticipation import get_index
from sklearn.preprocessing import MinMaxScaler


# C, for historical data


def get_data(dayback=1):
    get_data = get_index(dayback=dayback)  # C for current day dayback will be 0
    dfFii = get_data[1]
    df1 = get_data[0]

    # df1['date'] = df1['date'].dt.strftime('%Y%m%dT')
    # print(df1.columns.tolist())
    df_hist = pd.read_csv('D:/demos/nfAllHist2.csv')

    # df_hist['date'] = df_hist['date'].dt.strftime('%Y%m%dT')
    # print(custom_tabulate(dfmerge))

    dfhist_columns = df_hist.columns.tolist()
    # print(dfhist_columns)
    df1 = df1[dfhist_columns]
    # print(df1.columns.tolist())
    df1['date']= pd.to_datetime(df1['date'], format="%Y-%m-%d")
    df1['date'] = df1['date'].dt.strftime("%Y%m%dT")

    # C number of days displayed on chart
    if dayback == 0:
        dfmerge = pd.concat([df_hist, df1])
        dfmerge = dfmerge.reset_index(drop=True)
        dfmerge.to_csv('d:/demos/nfAllHist2.csv', index=False, mode='w')
    else:
        dfmerge = df_hist.copy()
        dfmerge = dfmerge.reset_index(drop=True)

    return dfmerge, dfFii


def get_chart(dayback=0):
    # dayback = 1

    dfmergelist = get_data(dayback=dayback)
    dfmerge = dfmergelist[0]
    dffFii = dfmergelist[1]
    show_last_days = 200
    dfmerge = dfmerge.copy().tail(show_last_days)  # Just need to visualize last 100 days trend
    dfmerge = dfmerge.set_index(dfmerge['date'], drop=True)
    dfmerge = dfmerge.drop(['date', 'symbol'], axis=1)

    def remove_commas(value):
        if isinstance(value, str):
            return value.replace(',', '')
        return value

    # Apply the function to each element in the DataFrame
    dfmerge = dfmerge.map(remove_commas)

    # Convert columns to numeric (optional)
    dfmerge = dfmerge.apply(pd.to_numeric, errors='ignore')

    # dfmerge = dfmerge.astype(float)
    # print(custom_tabulate(dfmerge))
    # dfmerge = df_hist.copy()
    dfmerge['avg_price'] = ((dfmerge.iloc[0]['open']) + (dfmerge.iloc[0]['high']) +
                            (dfmerge.iloc[0]['low']) + (dfmerge.iloc[0]['close'])) / 4
    dfmerge['ad_ratio'] = dfmerge['adv'] / dfmerge['dec']
    dfmerge['hl_ratio'] = dfmerge['new_high'] / dfmerge['new_low']
    scaler = MinMaxScaler(feature_range=(0, 100))
    dfmerge[['ad_ratio', 'hl_ratio']] = scaler.fit_transform(dfmerge[['ad_ratio', 'hl_ratio']])

    # text2 = (f"<br>Nifty % Chg: {nifty_return:.2f}%<br>Delivery % of stocks > 10L volume: {delvery_per:.2f}%<br>"
    #          f"Nifty from 20SMA: {NFsma_20d_percent_diff:.2f}%<br>Volume from it's 20SMA: {
    #          NFsma_20d_percent_diff_qty:.2f}%"
    #          f"<br>.............<br>FII Fut Idx OI Chg: {indexoi_fu_chg:.2f}<br>FII Option Idx OI Chg: {
    #          indexoi_op_chg:.2f}"
    #          f"<br>FII Idx Fu Buy-Sell: {index_fu_buy_sell:.2f}<br>FII Idx Opt Buy-Sell: {index_op_buy_sell:.2f}")

    # delvery_per = dfmerge.iloc[-1]['per_delivery']

    # dfmerge['ad_ratio'] = dfmerge['adv'] / dfmerge['dec']
    # dfmerge['hl_ratio'] = dfmerge['new_high'] / dfmerge['new_low']
    # indexoi_fu_chg = (100*((dfmerge.iloc[-1]['FIIindex_fu_oi_val']) - (dfmerge.iloc[-2]['FIIindex_fu_oi_val']))/
    #                   (dfmerge.iloc[-2]['FIIindex_fu_oi_val']))
    # indexoi_op_chg = (100*((dfmerge.iloc[-1]['FIIoptions_op_oi_val']) - (dfmerge.iloc[-2]['FIIoptions_op_oi_val']))/
    #                   (dfmerge.iloc[-2]['FIIoptions_op_oi_val']))
    # index_fu_buy_sell = (dfmerge.iloc[-1]['FIIindex_fu_buy_sell'])
    # index_op_buy_sell = (dfmerge.iloc[-1]['FIIoptions_op_buy_sell'])
    # nifty_return = 100*((dfmerge.iloc[-1]['close'])-(dfmerge.iloc[-2]['close']))/(dfmerge.iloc[-2]['close'])
    # dfmerge['20sma'] = dfmerge['close'].rolling(20).mean()
    # NFsma_20d_percent_diff = 100*((dfmerge.iloc[-1]['close']) - dfmerge.iloc[-1]['20sma'])/dfmerge.iloc[-1]['20sma']
    # dfmerge['20smaqty'] = dfmerge['qty'].astype().rolling(20).mean()
    # NFsma_20d_percent_diff_qty = 100*(dfmerge.iloc[-1]['qty'] - dfmerge.iloc[-1]['20smaqty'])/dfmerge.iloc[-1]['20smaqty']

# C *************************************************** FII ***************************************************


    # fig = {}

    num_rows = 7
    fig = make_subplots(rows=num_rows, cols=3, shared_xaxes=True, start_cell="top-left",
                        subplot_titles=('', ''),
                        row_heights=[0.40, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10], vertical_spacing=0.03,
                        horizontal_spacing=0.03,
                        specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}],
                               [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}],
                               [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}],
                               [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}],
                               [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}],
                               [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}],
                               [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}],
                               ])

    # fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Candlestick(x=dfmerge.index, open=dfmerge['open'], high=dfmerge['high'],
                                 low=dfmerge['low'],
                                 close=dfmerge['close'],legendgroup='candlestick',
                                 showlegend=False,
                                 name='<span style="color:white; font-weight:bold">Nifty</span>'),
                  col=1, row=1)  # candlestick
    fig.add_trace(go.Scatter(x=dfmerge.index, y=dfmerge['ad_ratio'], mode='lines',
                             name='<span style="color:magenta">Advance-Decline ratio</span>',
                             line=dict(width=1),
                             marker=dict(color='magenta'), showlegend=False), secondary_y=True, col=1, row=1)

    fig.add_trace(
        go.Scatter(
            x=dfmerge.index,
            y=dfmerge['hl_ratio'],
            mode='lines',
            name=f'<span style="color:cyan">New 52WK H and L ratio</span>',
            line=dict(width=1),
            marker=dict(color='cyan'), showlegend=False,
        ),
        secondary_y=True, col=1, row=1)
    # print(dfmerge.columns)
    FII_cols = [
        'fiiIdxFutlong', 'fiiIdxFutshort', 'fiiIdxCEshort', 'fiiIdxPEshort',
        'fiiIdxCElong', 'fiiIdxPElong',
        'fiiStkFutlong', 'fiiStkFutshort', 'fiiStkCEshort', 'fiiStkPEshort',
        'fiiStkCElong', 'fiiStkPElong'
    ]

    # print(custom_tabulate(dfmerge[FII_cols]))
    # for i in range(0, int(len(FII_cols)-1), 2):
    #     print(FII_cols[i], FII_cols[i+1])
    fii_name_list = [
        'fiiIdxFutlong', 'fiiIdxFutshort', 'fiiIdxCEshort', 'fiiIdxPEshort',
        'fiiIdxCElong', 'fiiIdxPElong',
        'fiiStkFutlong', 'fiiStkFutshort', 'fiiStkCEshort', 'fiiStkPEshort',
        'fiiStkCElong', 'fiiStkPElong'
    ]
    fii_color_list = ['limegreen', 'OrangeRed', 'coral', 'skyblue', 'cyan', 'orange',
                      'limegreen', 'OrangeRed', 'coral', 'skyblue', 'cyan', 'orange']
    k = -1

    # for i in range(0,int(len(FII_cols)-1),2):
    #     print(fii_color_list[i],fii_color_list[i+1] )

    for i in range(0, int(len(FII_cols) - 1), 2):
        k += 1
        fig.add_trace(
            go.Scatter(
                x=dfmerge.index,
                y=dfmerge[FII_cols[i]],
                mode='lines+markers',
                name=f'<span style="color:{fii_color_list[i]}">{fii_name_list[i]}</span>',
                line=dict(width=1),
                marker=dict(size=3,
                            color=fii_color_list[i],
                            opacity=np.round(dfmerge[FII_cols[i]] / dfmerge[FII_cols[i]].max(), 4),
                            ), showlegend=False,
            ),
            secondary_y=False, col=1, row=2 + k)

        fig.add_trace(
            go.Scatter(
                x=dfmerge.index,
                y=dfmerge[FII_cols[i + 1]],
                mode='lines+markers',
                name=f'<span style="color:{fii_color_list[i + 1]}">{fii_name_list[i + 1]}</span>',
                line=dict(width=1),
                marker=dict(size=3,
                            color=fii_color_list[i + 1],
                            opacity=np.round(dfmerge[FII_cols[i + 1]] / dfmerge[FII_cols[i + 1]].max(), 4),
                            ), showlegend=False,
            ),
            secondary_y=False, col=1, row=2 + k)

    title_color = [
        '<span style="color: limegreen;">Fii Index FutLong</span>'
        '<span style="color: OrangeRed;"> vs FutShort &#8595;</span>',
        '<span style="color: coral;">Fii Index CE short</span>'
        '<span style="color: skyblue;"> vs PE Short &#8595;</span>',
        '<span style="color: cyan;">FII Index CE Long</span>'
        '<span style="color: orange;"> vs PE Long &#8595;</span>',
        '<span style="color: limegreen;">FII Stock Fut Long</span>'
        '<span style="color: OrangeRed;"> vs FutShort &#8595;</span>',
        '<span style="color: coral;">FII Stock CE Short</span>'
        '<span style="color: skyblue;"> vs PE Short &#8595;</span>',
        '<span style="color: cyan;">FII Stock CE Long</span>'
        '<span style="color: orange;"> vs PE Long &#8595;</span>',
        '',''
    ]
    len(title_color)
    for row in range(0, num_rows + 1):
        # print(row)
        fig.update_yaxes(showgrid=False, color='white', col=1, row=row + 1, tickfont=dict(size=8),
                         zeroline=False, secondary_y=True)
        fig.update_yaxes(showgrid=False, color='white', col=1, row=row + 1, tickfont=dict(size=8),
                         zeroline=False, secondary_y=False)
        fig.update_xaxes(title = title_color[row],showgrid=False, showline=False, color='black', col=1,
                         row=row + 1,title_font=dict(size=8))

    fig.update_layout(yaxis_tickformat='d')  # d for showing full digits else it will show 19000 as 19k

    fig["layout"]["xaxis"]["rangeslider"]["visible"] = False

    title_y = ['&#8593; Nifty vs <span style="color: cyan;"> New 52 wk H/L ratio</span>'
               '<span style="color: magenta;"> and adv/dec ratio</span>']

    fig.update_yaxes(title = title_y[0],showline=False, showgrid=False,
                     col=1, row=1, secondary_y=True, range=[0, 300], title_font=dict(size=10),
                     tickfont=dict(size=8,color='black'))

    fig.update_xaxes(showline=True, color='white', showgrid=False, row=1, col=1)
    fig.update_xaxes(showline=True, color='white', showgrid=False, row=4, col=1)

    fig.update_xaxes(showline=True, color='white', showgrid=False,tickfont=dict(size=8),
                        type='category', tickangle=90, row=7, col=1)


# C ************************************************************************** DII ****************************


    dfmerge['fii_index_net'] = (dfmerge['fiiIdxFutlong'] - dfmerge['fiiIdxFutshort'])
    dfmerge['fii_index_delta'] = (dfmerge['fii_index_net'] - dfmerge['fii_index_net'].shift(1)).cumsum()
    dfmerge['dii_index_net'] = (dfmerge['diiIdxFutlong'] - dfmerge['diiIdxFutshort'])
    dfmerge['dii_index_delta'] = (dfmerge['dii_index_net'] - dfmerge['dii_index_net'].shift(1)).cumsum()

    scaler = MinMaxScaler(feature_range=(0, 100))
    dfmerge[['fii_index_delta', 'dii_index_delta']] = scaler.fit_transform(dfmerge[['fii_index_delta', 'dii_index_delta']])

    dfmerge = dfmerge.fillna(0.001)

    fig.add_trace(go.Scatter(x=dfmerge.index, y=dfmerge['close'], mode='lines',
                             line=dict(width=1),
                             marker=dict(color='white'), showlegend=False), secondary_y=False, col=2, row=1)

    fig.add_trace(
        go.Scatter(
            x=dfmerge.index,
            y=dfmerge['fii_index_delta'],
            mode='lines+markers',
            line=dict(width=1),
            marker=dict(color='cyan',size=4), showlegend=False,
        ),
        secondary_y=True, col=2, row=1)
    fig.add_trace(
        go.Scatter(
            x=dfmerge.index,
            y=dfmerge['dii_index_delta'],
            mode='lines+markers',
            line=dict(width=1),
            marker=dict(color='limegreen',size=4), showlegend=False,
        ),
        secondary_y=True, col=2, row=1)

    # print(dfmerge.columns)
    DII_cols = [
        'diiIdxFutlong', 'diiIdxFutshort', 'diiIdxCEshort', 'diiIdxPEshort',
        'diiIdxCElong', 'diiIdxPElong',
        'diiStkFutlong', 'diiStkFutshort', 'diiStkCEshort', 'diiStkPEshort',
        'diiStkCElong', 'diiStkPElong'
    ]

    # print(custom_tabulate(dfmerge[DII_cols]))
    # for i in range(0, int(len(DII_cols)-1), 2):
    #     print(DII_cols[i], DII_cols[i+1])
    dii_name_list = [
        'diiIdxFutlong', 'diiIdxFutshort', 'diiIdxCEshort', 'diiIdxPEshort',
        'diiIdxCElong', 'diiIdxPElong',
        'diiStkFutlong', 'diiStkFutshort', 'diiStkCEshort', 'diiStkPEshort',
        'diiStkCElong', 'diiStkPElong'
    ]
    dii_color_list = ['limegreen', 'OrangeRed', 'coral', 'skyblue', 'cyan', 'orange',
                      'limegreen', 'OrangeRed', 'coral', 'skyblue', 'cyan', 'orange']
    k = -1

    # for i in range(0,int(len(DII_cols)-1),2):
    #     print(dii_color_list[i],dii_color_list[i+1] )


    for i in range(0, int(len(DII_cols) - 1), 2):
        k += 1
        if (dfmerge[DII_cols[i]] != 0.001).any() and (dfmerge[DII_cols[i]] != 0.0).any():  # Option writing is 0 for DII
            opacity_value = np.round(dfmerge[DII_cols[i]] / dfmerge[DII_cols[i]].max(), 4)
        else:
            opacity_value=0
        if (dfmerge[DII_cols[i + 1]] != 0.001).any() and (
                dfmerge[DII_cols[i + 1]] != 0.0).any():  # Option writing is 0 for DII
            opacity_value2 = np.round(dfmerge[DII_cols[i + 1]] / dfmerge[DII_cols[i + 1 ]].max(), 4)
        else:
            opacity_value2 = 0

        fig.add_trace(
            go.Scatter(
                x=dfmerge.index,
                y=dfmerge[DII_cols[i]],
                mode='lines+markers',
                name=f'<span style="color:{dii_color_list[i]}">{dii_name_list[i]}</span>',
                line=dict(width=1),
                marker=dict(size=3,
                            color=dii_color_list[i],
                            opacity=opacity_value,
                            ), showlegend=False,
            ),
            secondary_y=False, col=2, row=2 + k)

        fig.add_trace(
            go.Scatter(
                x=dfmerge.index,
                y=dfmerge[DII_cols[i + 1]],
                mode='lines+markers',
                name=f'<span style="color:{dii_color_list[i + 1]}">{dii_name_list[i + 1]}</span>',
                line=dict(width=1),
                marker=dict(size=3,
                            color=dii_color_list[i + 1],
                            opacity=opacity_value2,
                            ), showlegend=False,
            ),
            secondary_y=False, col=2, row=2 + k)


    dii_title_color = [
        '<span style="color: limegreen;">Dii Index FutLong</span>'
        '<span style="color: OrangeRed;"> vs FutShort &#8595;</span>',
        '<span style="color: coral;">Dii Index CE short</span>'
        '<span style="color: skyblue;"> vs PE Short &#8595;</span>',
        '<span style="color: cyan;">DII Index CE Long</span>'
        '<span style="color: orange;"> vs PE Long &#8595;</span>',
        '<span style="color: limegreen;">DII Stock Fut Long</span>'
        '<span style="color: OrangeRed;"> vs FutShort &#8595;</span>',
        '<span style="color: coral;">DII Stock CE Short</span>'
        '<span style="color: skyblue;"> vs PE Short &#8595;</span>',
        '<span style="color: cyan;">DII Stock CE Long</span>'
        '<span style="color: orange;"> vs PE Long &#8595;</span>',
        '',''
    ]
    for row in range(0, num_rows + 1):
        # row = 1
        # print(row)
        fig.update_yaxes(showgrid=False, color='white', col=2, row=row + 1, tickfont=dict(size=8),
                         zeroline=False, secondary_y=True)
        fig.update_yaxes(showgrid=False, color='white', col=2, row=row + 1, tickfont=dict(size=8),
                         zeroline=False, secondary_y=False)
        fig.update_xaxes(title = dii_title_color[row],showgrid=False, showline=False, color='black', col=2,
                         row=row + 1,title_font=dict(size=8))

    # fig.update_layout(yaxis_tickformat='d')  # d for showing full digits else it will show 19000 as 19k

    fig["layout"]["xaxis"]["rangeslider"]["visible"] = False

    title_y = ['&#8593; Nifty vs <span style="color: cyan;">FII Index Fut daily chg over time</span>'
               '<span style="color: limegreen;"> vs DII </span>']

    fig.update_yaxes(title = title_y[0],showline=False, showgrid=False,
                     col=2, row=1, secondary_y=True, title_font=dict(size=10),
                     tickfont=dict(size=8,color='black'))
    fig.update_yaxes(showline=False, showgrid=False,
                     col=2, row=1, secondary_y=False,
                     tickfont=dict(size=8,color='black'))
    fig.update_xaxes(showline=True, color='white', showgrid=False, row=1, col=2)

    fig.update_xaxes(showline=True, color='white', showgrid=False, row=4, col=2)

    fig.update_xaxes(showline=True, color='white', showgrid=False,tickfont=dict(size=8),
                        type='category', tickangle=90, row=7, col=2)

# C ************************************************************************ Prop *****************************


    dfmerge['prop_index_net'] = (dfmerge['propIdxFutlong'] - dfmerge['propIdxFutshort'])
    dfmerge['prop_index_delta'] = (dfmerge['prop_index_net'] - dfmerge['prop_index_net'].shift(1)).cumsum()

    scaler = MinMaxScaler(feature_range=(0, 100))
    dfmerge[['prop_index_delta', 'dii_index_delta']] = scaler.fit_transform(dfmerge[['prop_index_delta', 'dii_index_delta']])

    dfmerge = dfmerge.fillna(0.001)

    fig.add_trace(go.Scatter(x=dfmerge.index, y=dfmerge['close'], mode='lines',
                             line=dict(width=1),
                             marker=dict(color='white'), showlegend=False), secondary_y=False, col=3, row=1)

    fig.add_trace(
        go.Scatter(
            x=dfmerge.index,
            y=dfmerge['prop_index_delta'],
            mode='lines+markers',
            line=dict(width=1),
            marker=dict(color='cyan',size=4), showlegend=False,
        ),
        secondary_y=True, col=3, row=1)
    fig.add_trace(
        go.Scatter(
            x=dfmerge.index,
            y=dfmerge['dii_index_delta'],
            mode='lines+markers',
            line=dict(width=1),
            marker=dict(color='limegreen',size=4), showlegend=False,
        ),
        secondary_y=True, col=3, row=1)

    # print(dfmerge.columns)
    Prop_cols = [
        'propIdxFutlong', 'propIdxFutshort', 'propIdxCEshort', 'propIdxPEshort',
        'propIdxCElong', 'propIdxPElong',
        'propStkFutlong', 'propStkFutshort', 'propStkCEshort', 'propStkPEshort',
        'propStkCElong', 'propStkPElong'
    ]

    # print(custom_tabulate(dfmerge[DII_cols]))
    # for i in range(0, int(len(DII_cols)-1), 2):
    #     print(DII_cols[i], DII_cols[i+1])
    prop_name_list = [
        'propIdxFutlong', 'propIdxFutshort', 'propIdxCEshort', 'propIdxPEshort',
        'propIdxCElong', 'propIdxPElong',
        'propStkFutlong', 'propStkFutshort', 'propStkCEshort', 'propStkPEshort',
        'propStkCElong', 'propStkPElong'
    ]
    prop_color_list = ['limegreen', 'OrangeRed', 'coral', 'skyblue', 'cyan', 'orange',
                      'limegreen', 'OrangeRed', 'coral', 'skyblue', 'cyan', 'orange']
    k = -1

    # for i in range(0,int(len(DII_cols)-1),2):
    #     print(prop_color_list[i],prop_color_list[i+1] )


    for i in range(0, int(len(Prop_cols) - 1), 2):
        k += 1
        if (dfmerge[Prop_cols[i]] != 0.001).any() and (dfmerge[Prop_cols[i]] != 0.0).any():  # Option writing is 0 for DII
            opacity_value = np.round(dfmerge[Prop_cols[i]] / dfmerge[Prop_cols[i]].max(), 4)
        else:
            opacity_value=0
        if (dfmerge[Prop_cols[i + 1]] != 0.001).any() and (
                dfmerge[Prop_cols[i + 1]] != 0.0).any():  # Option writing is 0 for DII
            opacity_value2 = np.round(dfmerge[Prop_cols[i + 1]] / dfmerge[Prop_cols[i + 1 ]].max(), 4)
        else:
            opacity_value2 = 0

        fig.add_trace(
            go.Scatter(
                x=dfmerge.index,
                y=dfmerge[Prop_cols[i]],
                mode='lines+markers',
                name=f'<span style="color:{prop_color_list[i]}">{prop_name_list[i]}</span>',
                line=dict(width=1),
                marker=dict(size=3,
                            color=prop_color_list[i],
                            opacity=opacity_value,
                            ), showlegend=False,
            ),
            secondary_y=False, col=3, row=2 + k)

        fig.add_trace(
            go.Scatter(
                x=dfmerge.index,
                y=dfmerge[Prop_cols[i + 1]],
                mode='lines+markers',
                name=f'<span style="color:{prop_color_list[i + 1]}">{prop_name_list[i + 1]}</span>',
                line=dict(width=1),
                marker=dict(size=3,
                            color=prop_color_list[i + 1],
                            opacity=opacity_value2,
                            ), showlegend=False,
            ),
            secondary_y=False, col=3, row=2 + k)


    prop_title_color = [
        '<span style="color: limegreen;">Prop Index FutLong</span>'
        '<span style="color: OrangeRed;"> vs FutShort &#8595;</span>',
        '<span style="color: coral;">Prop Index CE short</span>'
        '<span style="color: skyblue;"> vs PE Short &#8595;</span>',
        '<span style="color: cyan;">Prop Index CE Long</span>'
        '<span style="color: orange;"> vs PE Long &#8595;</span>',
        '<span style="color: limegreen;">Prop Stock Fut Long</span>'
        '<span style="color: OrangeRed;"> vs FutShort &#8595;</span>',
        '<span style="color: coral;">Prop Stock CE Short</span>'
        '<span style="color: skyblue;"> vs PE Short &#8595;</span>',
        '<span style="color: cyan;">Prop Stock CE Long</span>'
        '<span style="color: orange;"> vs PE Long &#8595;</span>',
        '',''
    ]
    for row in range(0, num_rows + 1):
        # row = 1
        # print(row)
        fig.update_yaxes(showgrid=False, color='white', col=3, row=row + 1, tickfont=dict(size=8),
                         zeroline=False, secondary_y=True)
        fig.update_yaxes(showgrid=False, color='white', col=3, row=row + 1, tickfont=dict(size=8),
                         zeroline=False, secondary_y=False)
        fig.update_xaxes(title = prop_title_color[row],showgrid=False, showline=False, color='black', col=3,
                         row=row + 1,title_font=dict(size=8))

    # fig.update_layout(yaxis_tickformat='d')  # d for showing full digits else it will show 19000 as 19k

    fig["layout"]["xaxis"]["rangeslider"]["visible"] = False

    title_y = ['&#8593; Nifty vs <span style="color: cyan;">Prop Index Fut daily chg over time</span>'
               '<span style="color: limegreen;"> vs DII </span>']

    fig.update_yaxes(title = title_y[0],showline=False, showgrid=False,
                     col=3, row=1, secondary_y=True, title_font=dict(size=10),
                     tickfont=dict(size=8,color='black'))
    fig.update_yaxes(showline=False, showgrid=False,
                     col=3, row=1, secondary_y=False,
                     tickfont=dict(size=8,color='black'))
    fig.update_xaxes(showline=True, color='white', showgrid=False, row=1, col=3)

    fig.update_xaxes(showline=True, color='white', showgrid=False, row=4, col=3)

    fig.update_xaxes(showline=True, color='white', showgrid=False,tickfont=dict(size=8),
                        type='category', tickangle=90, row=7, col=3)


# C ***************************************************** Final Output *********************************

    monitor_index = 2
    monitors = get_monitors()
    chosen_monitor = monitors[monitor_index]
    screen_width = chosen_monitor.width
    screen_height = chosen_monitor.height

    fig.update_layout(paper_bgcolor='black', plot_bgcolor='black', height=int(screen_height * 0.7),
                      width=int(screen_width * 0.8),
                      title='<a href="https://www.youtube.com/@quantext/featured">Youtube @quantext</a>'
                            '<br>Participation wise Market Open interest',
                      showlegend=True, title_font=dict(size=12, color='white'),
                      uniformtext=dict(minsize=11, mode='hide'),
                      margin=dict(t=65, l=5, r=5, b=45)
                      )

    # fig.layout.xaxis.type = 'category'

    plot(fig, auto_open=True)
    # fig.show()  # Uncomment this if you want to run inside Pycharm or ipython

get_chart(dayback=1)
