import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

root = 'd:/demos/'
df = pd.read_csv(root + 'nifty_index_data.csv')
print(df.head(4).to_dict())
# c = df.iloc[-1]['close']
df = df.set_index(df['date'], drop=True)
dfcandle = df.copy()[['open', 'high', 'low', 'close']]
print(df.columns.tolist())
# C day 2
for i in range(0, 10):
    # i = 1
    df['return' + str(i)] = ((df['close'].shift(i) - df['close'].shift(i + 1)) / df['close'].shift(i)).multiply(100)
    df['qty_return' + str(i)] = ((df['qty'].shift(i) - df['qty'].shift(i + 1)) / df['qty'].shift(i)).multiply(100)
    df['trades_return' + str(i)] = (
                (df['trades'].shift(i) - df['trades'].shift(i + 1)) / df['trades'].shift(i)).multiply(100)
    df['avgprice_return' + str(i)] = (
                (df['avg_price'].shift(i) - df['avg_price'].shift(i + 1)) / df['avg_price'].shift(i)).multiply(100)
    df['high_return' + str(i)] = ((df['high'].shift(i) - df['high'].shift(i + 1)) / df['high'].shift(i)).multiply(100)
    df['low_return' + str(i)] = ((df['low'].shift(i) - df['low'].shift(i + 1)) / df['low'].shift(i)).multiply(100)
    df['gap_return' + str(i)] = ((df['open'].shift(i) - df['close'].shift(i + 1)) / df['close'].shift(i + 1)).multiply(
        100)
    df['hl_ratio' + str(i)] = ((df['hl_ratio'].shift(i) - df['hl_ratio'].shift(i + 1)) / df['hl_ratio'].shift(i + 1)).multiply(
        100)
    df['ad_ratio' + str(i)] = ((df['ad_ratio'].shift(i) - df['ad_ratio'].shift(i + 1)) / df['ad_ratio'].shift(i + 1)).multiply(
        100)



df = df.dropna()
df['Target'] = np.where(df['close'].shift(-1) >= df['close'], 1, -1)

# df['ad_ratio'] = df['adv']/df['dec']
# df['hl_ratio'] = df['new_high']/df['new_low']

# print(df.columns.tolist())
# C note on this version we lso deleted adv decline nd new high new low features, 1st we try with OHLC data for baseline
df = df.drop(['symbol', 'date', 'adv', 'dec', 'open', 'high', 'low', 'close', 'qty', 'trades', 'new_high',
              'new_low', 'avg_price', 'ad_ratio', 'hl_ratio'], axis=1)

features = list(df.columns[:90])
# features = list(df.columns[:13])
print(features)
print(df.head())


from sklearn.model_selection import train_test_split

y = df["Target"]
x = df[features]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.50, shuffle=False)

rfdt = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=5, min_samples_split=40,
                              min_weight_fraction_leaf=0.01, random_state=99)
class_names = [str(class_label) for class_label in set(y_train)]

rfdt = rfdt.fit(x_train, y_train)

y_pred = rfdt.predict(x_test)
acc2 = accuracy_score(y_test, y_pred) * 100
print(acc2)

plt.figure(figsize=(12, 8))
plot_tree(rfdt.estimators_[2], feature_names=x_test.columns, class_names=class_names, filled=True)
plt.show()

imp_rf = pd.DataFrame(rfdt.feature_importances_, columns=['Imp'], index=x.columns).sort_values(by='Imp',
                                                                                               ascending=False)  # feature importance score
print(imp_rf)
list_imp_features = imp_rf.index.tolist()
print(list_imp_features)

# C using  most important features for simple decision tree : Spoiler, doesn't work, since no bagging and bootstrapping
# imp_rf_filter = imp_rf[imp_rf['Imp'] != 0]
# mod_features = imp_rf_filter.index.tolist()
# x_train2 = x_train[mod_features]
# x_test2 = x_test[mod_features]
# x_train2.columns.tolist()
# x_test2.columns.tolist()
# #
# dt2 = dt.fit(x_train2, y_train)
# y_pred2 = dt2.predict(x_test2)
# acc2 = accuracy_score(y_test, y_pred2) * 100
# print(acc2)

bias_test = pd.DataFrame(y_pred)
bias_test = bias_test.set_index(y_test.index)
bias_test.columns = ['bias_test']
probabilities = rfdt.predict_proba(x_test)
probabilities_df = pd.DataFrame(probabilities, columns=['probability_bearish', 'probability_bullish'])
probabilities_df = probabilities_df.set_index(x_test.index, drop=True)
print(x_test.columns.tolist())

# x_test = x_test.drop(['return0', 'qty_return0', 'trades_return0', 'avgprice_return0', 'high_return0', 'low_return0',
#                       'gap_return0', 'return1', 'qty_return1', 'trades_return1', 'avgprice_return1',
#                       'high_return1', 'low_return1', 'gap_return1', 'return2', 'qty_return2', 'trades_return2',
#                       'avgprice_return2', 'high_return2', 'low_return2', 'gap_return2', 'return3', 'qty_return3',
#                       'trades_return3', 'avgprice_return3', 'high_return3', 'low_return3', 'gap_return3', 'return4',
#                       'qty_return4', 'trades_return4', 'avgprice_return4', 'high_return4', 'low_return4',
#                       'gap_return4', 'return5', 'qty_return5', 'trades_return5', 'avgprice_return5', 'high_return5',
#                       'low_return5', 'gap_return5', 'return6', 'qty_return6', 'trades_return6', 'avgprice_return6',
#                       'high_return6', 'low_return6', 'gap_return6', 'return7', 'qty_return7', 'trades_return7',
#                       'avgprice_return7', 'high_return7', 'low_return7', 'gap_return7', 'return8', 'qty_return8',
#                       'trades_return8', 'avgprice_return8', 'high_return8', 'low_return8', 'gap_return8', 'return9',
#                       'qty_return9', 'trades_return9', 'avgprice_return9', 'high_return9', 'low_return9', 'gap_return9'], axis=1)

# result_df = pd.concat([x_test, probabilities_df], axis=1)
# C y_test are actual results, y_pred is bias and next 2 are bias ion probabilities terms

result_df = pd.concat([y_test, bias_test, probabilities_df], axis=1)
# result_df = result_df.set_index()
# C For plotting

result_df['Actual_Plus'] = np.where(result_df['Target'] == 1, 1, 0)
result_df['Actual_Minus'] = np.where(result_df['Target'] == -1, 1, 0)

result_df['prob60_Long'] = np.where(result_df['probability_bullish'] >= 0.60, 1, 0)
result_df['buy_pred'] = np.where(result_df['bias_test'] >= 1, 1, 0)
result_df['correct_long'] = np.where(np.logical_and(result_df['bias_test'] == 1, result_df['Target'] == 1), 1, 0)
result_df['false_long'] = np.where(np.logical_and(result_df['bias_test'] == 1, result_df['Target'] == -1), 1, 0)
result_df['correct_long_60'] = np.where(np.logical_and(result_df['prob60_Long'] == 1, result_df['Target'] == 1), 1, 0)
result_df['false_long_60'] = np.where(np.logical_and(result_df['prob60_Long'] == 1, result_df['Target'] == -1), 1, 0)

result_df['prob60_SH'] = np.where(result_df['probability_bearish'] >= 0.60, 1, 0)
result_df['sell_pred'] = np.where(result_df['bias_test'] == -1, 1, 0)
result_df['correct_short'] = np.where(np.logical_and(result_df['bias_test'] == -1, result_df['Target'] == -1), 1, 0)
result_df['false_short'] = np.where(np.logical_and(result_df['bias_test'] == -1, result_df['Target'] == 1), 1, 0)
result_df['correct_short_60'] = np.where(np.logical_and(result_df['prob60_SH'] == 1, result_df['Target'] == -1), 1, 0)
result_df['false_short_60'] = np.where(np.logical_and(result_df['prob60_SH'] == 1, result_df['Target'] == 1), 1, 0)

total_plus = result_df['Actual_Plus'].sum()
correct_long = result_df['correct_long'].sum()
false_long = result_df['false_long'].sum()
long_success = correct_long - false_long

total_minus = result_df['Actual_Minus'].sum()
correct_short = result_df['correct_short'].sum()
false_short = result_df['false_short'].sum()
short_success = correct_short - false_short
percent_profit_long = 100 * (correct_long / (correct_long + false_long))
percent_profit_short = 100 * (correct_short / (correct_short + false_short))
averageprofit_per = (percent_profit_long + percent_profit_short) / 2
# C............ only > 60 % prob .............................
# C Long
correct_long_60 = result_df['correct_long_60'].sum()
false_long_60 = result_df['false_long_60'].sum()
long_success_60 = correct_long_60 - false_long_60
long_success_per_60 = 100 * (correct_long_60 / (correct_long_60 + false_long_60))
# C Short
correct_short_60 = result_df['correct_short_60'].sum()
false_short_60 = result_df['false_short_60'].sum()
short_success_60 = correct_short_60 - false_short_60
short_success_per_60 = 100 * correct_short_60 / (correct_short_60 + false_short_60)

total_buybias = result_df['buy_pred'].sum()

total_sellbias = result_df['sell_pred'].sum()
strong_bias_bearish = 100 * (result_df['prob60_SH'].sum() / total_sellbias)

print(f"Total Plus: {total_plus}, Correct Long: {correct_long}, False Long: {false_long},"
      f" Long Success: {long_success}, "
      f"Total Minus: {total_minus},"
      f"Correct Short: {correct_short}, False Short: {false_short},"
      f"Short Success: {short_success},"
      f"Percent Profit Long: {percent_profit_long:.2f}%, "
      f"Percent Profit Short: {percent_profit_short:.2f}%, Average Profit Percentage: {averageprofit_per:.2f}%")


print(f"Correct Long 60: {correct_long_60}, False Long 60: {false_long_60}, Long Success 60: {long_success_60},"
      f" Long Success Percentage 60: {long_success_per_60:.2f}%")

print(f"Correct Short 60: {correct_short_60}, False Short 60: {false_short_60}, Short Success 60: {short_success_60},"
      f" Short Success Percentage 60: {short_success_per_60:.2f}%")



# C ................................................................. BackTest ......................
dfcandle_test = dfcandle.copy()[len(result_df):]
print(dfcandle_test[0:12].index.tolist())
print(result_df[0:12].index.tolist())
date_start_test = result_df.index[0]
dfcandle_test2 = dfcandle_test.loc[date_start_test:]

data = pd.concat([result_df, dfcandle_test2], axis=1)
data_bt = data[['Target', 'bias_test', 'close']]
# C backtest with closing values
# C long

slippage = 0.001
capital = 50000
stops = 0.01  # 1%
profit = 0.14  # 14% Almost no profit target
data_bt['slippage'] = data_bt['close'] * slippage
data_bt['ProfitTarget'] = data_bt['close'] * profit
data_bt['StopLoss'] = -data_bt['close'] * stops
data_bt['Capital'] = capital

data_bt['Long'] = np.where(data_bt['bias_test'] == 1, data_bt['close'].shift(-1) - data_bt['close'], 0)
data_bt['LongProfit_target'] = np.where(data_bt['Long'] > data_bt['ProfitTarget'], data_bt['ProfitTarget'],
                                        data_bt['Long'])
data_bt['Long_new'] = np.where(data_bt['Long'] < data_bt['StopLoss'], data_bt['StopLoss'], data_bt['LongProfit_target'])
data_bt['LongProfit'] = np.where(data_bt['Long_new'] > 0, data_bt['Long_new'] - data_bt['slippage'], 0)
data_bt['LongLoss'] = np.where(data_bt['Long_new'] < 0, data_bt['Long_new'] - data_bt['slippage'], 0)

# C short
data_bt['Short'] = np.where(data_bt['bias_test'] == -1, data_bt['close'] - data_bt['close'].shift(-1), 0)
data_bt['ShortProfit_target'] = np.where(data_bt['Short'] > data_bt['ProfitTarget'], data_bt['ProfitTarget'],
                                         data_bt['Short'])
data_bt['Short_new'] = np.where(data_bt['Short'] < data_bt['StopLoss'], data_bt['StopLoss'],
                                data_bt['ShortProfit_target'])
data_bt['ShortProfit'] = np.where(data_bt['Short_new'] > 0, data_bt['Short_new'] - data_bt['slippage'], 0)
data_bt['ShortLoss'] = np.where(data_bt['Short_new'] < 0, data_bt['Short_new'] - data_bt['slippage'], 0)

print(data_bt.columns.tolist())

net_long = data_bt['LongProfit'].sum()
net_short = data_bt['ShortProfit'].sum()
gross_profit = net_long + net_short
gross_loss = data_bt['LongLoss'].sum() + data_bt['ShortLoss'].sum()
net_total_profit = gross_profit - abs(gross_loss)

profit_factor = gross_profit / abs(gross_loss)
avg_profit_long = net_long / data['buy_pred'].sum()
avg_profit_short = net_short / data['sell_pred'].sum()
per_return = 100 * (net_total_profit / data_bt.iloc[-1]['Capital'])
period_tested = data.index[0] + '-' + data.index[-1]

data_bt['net_profit'] = data_bt['Long_new'] + data_bt['Short_new']
data_bt['color'] = np.where(data_bt['net_profit'] > 0, 'green', 'red')
data_bt['equity_curve'] = data_bt['net_profit'].cumsum()
data['close_diff'] = data['close'] - data['close'].shift()
data['equity_curve_underlying'] = data['close_diff'].cumsum()

data_bt['cummax'] = data_bt['equity_curve'].cummax()
# data_bt['Capital'] = 50000
# Calculate the drawdown as the percentage decline from the peak
data_bt['drawdown'] = (data_bt['equity_curve'] - data_bt['cummax'])
data_bt['drawdown%'] = 100 * (data_bt['drawdown'] / data_bt['Capital'])
data_bt['peak'] = data_bt['equity_curve'].cummax()
max_drawdown = data_bt['drawdown%'].min()

# Calculate Ulcer Index , it considers max dip in portfolio and time spent before recover, Lower the better
n = len(data_bt)
ulcer_index = (1 / n) * ((data_bt['drawdown'] / data_bt['peak']) ** 2).sum() ** 0.5 * 100

# Sharpe ratio
average_return = data_bt['net_profit'].mean()
std_dev_return = data_bt['net_profit'].std()

# Assume a risk-free rate (e.g., 0.01 for 1%)
risk_free_rate = 0.07

# C Calculate the Sharpe Ratio
sharpe_ratio = (average_return - risk_free_rate) / std_dev_return

# C Conservative winners and losers

# Identify consecutive winners and losers
data_bt['Winners'] = (data_bt['net_profit'] > 0).astype(int)
data_bt['Losers'] = (data_bt['net_profit'] < 0).astype(int)

# Calculate consecutive counts using a groupby approach
data_bt['ConsecutiveWinners'] = data_bt.groupby((data_bt['Winners'] != data_bt['Winners'].shift()).cumsum())[
    'Winners'].cumsum()
data_bt['ConsecutiveLosers'] = data_bt.groupby((data_bt['Losers'] != data_bt['Losers'].shift()).cumsum())[
    'Losers'].cumsum()
max_cons_winners = data_bt['ConsecutiveWinners'].max()
max_cons_losers = data_bt['ConsecutiveLosers'].max()

# C max profit %  and loss %
data_bt['return%'] = 100 * (data_bt['net_profit'] / data_bt['close'])
max_profit_per = data_bt['return%'].max()
max_loss_per = data_bt['return%'].min()

print(
    f'Period Tested: {period_tested},\nPercentage Return: {per_return:.2f}%,\nProfitable Trade%: {averageprofit_per:.2f}%,'
    f'\nAverage Profit/Trade: {average_return:.2f},\nprofit_factor: {profit_factor:.2f},'
    f'\nmax drawdown%: {max_drawdown:.2f}%,\nnet_total_profit: {net_total_profit:.2f},'
    f'\ngross_profit: {gross_profit:.2f},\ngross_loss: {gross_loss:.2f},\nmax_cons_winners : {max_cons_winners},'
    f'\nmax_cons_losers: {max_cons_losers},\nMax Profit%: {max_profit_per:.2f},\nMax Loss%: {max_loss_per:.2f}',
    f'\n\nNet Longs: {correct_long + false_long},'
    f'\n Net Correct Longs: {correct_long},\n Net False Longs:{false_long},'
    f'\n Long Profit%: {percent_profit_long:.2f}%,'
    f'\nNet Shorts: {correct_short + false_short},\n Net Correct Shorts: {correct_short},\n Net False Shorts:{false_short},'
    f'\nShort Profit%: {percent_profit_short:.2f}%,')

print(f'net_long: {net_long:.2f},\nnet_short: {net_short:.2f},')

print(f'\nUlcer Index: {ulcer_index:.2f}%, \nSharpe Ratio: {sharpe_ratio:.2f} ')
# C Equity curve

# data.to_csv('d:/demos/rf_depth.csv',index=None)

# C ........................................................plot for sanity check ..............


import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots

# rangeH = dfcandle_test2.tail(100)['high'].max()
# rangeL = dfcandle_test2.tail(100)['low'].min()
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.70, 0.30], vertical_spacing=0.003,
                    horizontal_spacing=0.0003,
                    specs=[[{"secondary_y": True}],
                           [{"secondary_y": True}]])
fig.add_trace(go.Candlestick(x=data.index, open=data['open'], high=data['high'],
                             low=data['low'], close=data['close'], name='<span style="color:white">Nifty</span>'),
              row=1, col=1)  # candlestick

fig.add_trace(go.Scatter(x=data_bt.index, y=data_bt['equity_curve'], mode='lines',
                         name='<span style="color:green">Equity Curve</span>', marker=dict(color='green')),
              secondary_y=True, row=1, col=1)

fig.add_trace(go.Scatter(x=data_bt.index, y=data_bt['drawdown%'], mode='lines',
                         name='<span style="color:blue">DrawDown%</span>', marker=dict(color='blue')),
              secondary_y=True, row=2, col=1)
fig.add_trace(go.Bar(x=data_bt.index, y=data_bt['return%'], name=f'<span style=f"color:white">Daily Returns%</span>',
                     marker=dict(color=data_bt['color'])),
              secondary_y=False, row=2, col=1)
fig.update_xaxes(showline=False, color='white', showgrid=False, type='category',
                 tickangle=90, zeroline=False, row=2)

fig.update_yaxes(showline=False, color='white', showgrid=False,
                 row=2)
fig.update_yaxes(showline=False, color='white', showgrid=False,
                 row=1)

fig.update_layout(paper_bgcolor='black', plot_bgcolor='black',
                  height=1070, width=1820, autosize=False,
                  xaxis=dict(showgrid=False, rangeslider=dict(visible=False)),
                  yaxis=dict(showgrid=False, tickformat='d'),
                  yaxis2=dict(showgrid=False, tickformat='d'))

plot(fig, auto_open=True)

