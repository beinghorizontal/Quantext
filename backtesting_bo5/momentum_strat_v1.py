from plotly.offline import plot
import plotly.graph_objects as go
import numpy as np
import yfinance as yf
from backtesting import Strategy, Backtest
from plotly.offline import plot

from bACKTEST_COMPARE import df_mp_daily, backtest

df = yf.download('^NSEI')
split = int(len(df)/2)
df_insample = df.head(split)
df_outsample = df.tail(split)

df_daily = df_insample.copy()
#df_daily = df_outsample.copy()

df_daily['High_5'] = df_daily['High'].rolling(5).max()

df_daily['AR'] = df_daily['High'] - df_daily['Low']
df_daily['AR_Long'] = df_daily['AR'].rolling(30).mean()
df_daily['AR_Small'] = df_daily['AR'].rolling(12).mean()

df_daily = df_daily.dropna()

long_condition  = ("(df_daily['High'] > df_daily['High_5'].shift(1)) &"
"(df_daily['AR_Small'] > df_daily['AR_Long'])")

df_daily['Long'] = np.where(eval(long_condition), 1, 0)

class BreakoutFive(Strategy):
    s1 = 22
    s2 = 12
    def init(self):
        self.entry_price_buy = None
    def next(self):
        pt_div = self.s1/100
        sl_div = self.s2/1000
        if not self.position:
            if self.data.Long[-1] == 1:
                self.buy()
                self.entry_price_buy = self.data.Close[-1]
        else:
            if self.position.is_long:
                if(self.data.High[-1] > self.entry_price_buy*(1+pt_div)
                or(self.data.Low[-1] < self.entry_price_buy *(1-sl_div))
                ):
                    self.position.close()


backtest = Backtest(df_daily, BreakoutFive, cash = 500_000, trade_on_close = True,
                    commission = 0.002)

stats = backtest.run()
print(stats)

opt_stats,heatmap = backtest.optimize(s1 = range(5,30),
                                      s2 = range(5,15),
                                      maximize='Sharpe Ratio',
                                      method = 'grid',
                                      constraint=lambda x: x.s1 > x.s2,
                                      return_heatmap=True,
                                      random_state=2)
print(opt_stats)

best_val1 = opt_stats._strategy.s1
print(f"The best s1 value is: {best_val1}")
best_val2 = opt_stats._strategy.s2
print(f"The best s2 value is: {best_val2}")

heatmap_df = heatmap.reset_index()
heatmap_df = heatmap_df.pivot(index='s1', columns='s2', values = 'Sharpe Ratio')

fig = go.Figure(data=go.Heatmap(z=heatmap_df.values, x=heatmap_df.columns,
                                y=heatmap_df.index, colorscale='viridis'))
fig.update_layout(title='Heatmap of Sharpe Ratio', xaxis_title='Stop Loss',
                  yaxis_title='Profit Target')
plot(fig)

