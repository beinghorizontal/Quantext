import pandas as pd
from scipy.stats import pointbiserialr
import plotly.express as px
from plotly.offline import plot

# C, for historical data


df = pd.read_csv('D:/demos/nfAllHist2.csv')

df = df.set_index(df['date'], drop=True)
df['avgprice'] = (df['open'] + df['high'] + df['low'] + df['close'])/4
df['ad_ratio'] = df['adv'] / df['dec']
df['hl_ratio'] = df['new_high'] / df['new_low']

df = df.drop(['date', 'symbol', 'open','high','low','close',
              'trades', 'adv', 'dec', 'new_high', 'new_low','diiIdxCEshort',
              'diiIdxPEshort','diiIdxCElong',
              'diiStkCElong', 'diiStkPElong','diiStkFutlong',
              'diiStkFutshort','propStkPElong','propStkCElong','propStkCEshort',
              'propStkPEshort', 'propStkFutshort','propStkFutlong',
              'fiiStkCEshort', 'fiiStkPEshort','fiiStkPElong','fiiStkCElong',
              ], axis=1)
def remove_commas(value):
    if isinstance(value, str):
        return value.replace(',', '')
    return value

# Apply the function to each element in the DataFrame
df = df.map(remove_commas)

# Convert columns to numeric (optional)
df = df.apply(pd.to_numeric, errors='ignore')

# df = df_hist.copy()

df = df.replace(0, 0.001)

df = df.pct_change()*100
# print(custom_tabulate(df))

df['target'] = (df['avgprice'].shift(-1) > 0).astype('int')
df = df.dropna()
# print(custom_tabulate(df[['avgprice','target']]))

continuous_columns = df.iloc[:, 0:23]
binary_column = df['target']

correlations = {}
for col in df.columns[:-1]:
    corr, pval = pointbiserialr(df[col], df['target'])
    correlations[col] = (corr, pval)

dfcor = pd.DataFrame(correlations)
dfcor = dfcor.transpose()
dfcor.columns = ['Correlation', 'P-Value']
dfcor['Combined'] = dfcor['Correlation'] * (1 - dfcor['P-Value'])  # Adjust weights as needed
dfcor['Variable'] = dfcor.index
dfcor = dfcor.reset_index(drop=True)

print("Point-Biserial Correlations:")
for col, (corr, pval) in correlations.items():
    print(f"- {col}: {corr:.3f} ({pval:.3f})")



fig = px.bar(dfcor, x='Variable', y='Combined', color='Correlation',
             text='Correlation', labels={'Combined': 'Combined Metric'},
             title='Combined Metric (Correlation * (1 - P-Value))')

# Customize layout
fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
fig.update_layout(xaxis_title='Variable', yaxis_title='Combined Metric', coloraxis_showscale=False)
plot(fig)

