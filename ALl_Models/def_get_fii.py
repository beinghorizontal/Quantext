import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import plot
from scipy.stats import pointbiserialr


# C, for historical data
def get_fii(visualize=False):
    # C get index data with FII participation
    df_fii = pd.read_csv('fii_dii.csv')

    df_fii = df_fii.set_index(df_fii['date'], drop=True)
    df_fiicandle = df_fii.copy()[['open', 'high', 'low', 'close']]

    def remove_commas(value):
        if isinstance(value, str):
            return value.replace(',', '')
        return value

    # Apply the function to each element in the DataFrame
    df_fii = df_fii.map(remove_commas)

    # Convert columns to numeric (optional)
    df_fii = df_fii.apply(pd.to_numeric, errors='ignore')

    # df_fii = df_fii_hist.copy()

    df_fii = df_fii.replace(0, 0.001)

    # C rough check if concat works and how many dates will get missed

    df_fii['avgprice'] = (df_fii['open'] + df_fii['high'] + df_fii['low'] + df_fii['close']) / 4
    "// Bullish"
    df_fii['adv_dec_per'] = df_fii['adv']/(df_fii['adv']+df_fii['dec'])
    df_fii['newH-newL_per'] = df_fii['new_high']/(df_fii['new_high']+df_fii['new_low'])

    df_fii['fiiIdxFut_Long'] = df_fii['fiiIdxFutlong'].pct_change() * 100
    df_fii['fiiIdxPE_Short'] = df_fii['fiiIdxPEshort'].pct_change() * 100
    df_fii['fiiIdxCE_Long'] = df_fii['fiiIdxCElong'].pct_change() * 100
    df_fii['fiiStkFut_Long'] = df_fii['fiiStkFutlong'].pct_change() * 100
    df_fii['diiIdxFut_Long'] = df_fii['diiIdxFutlong'].pct_change() * 100
    df_fii['propIdxFut_Long'] = df_fii['propIdxFutlong'].pct_change() * 100
    df_fii['propIdxPE_Short'] = df_fii['propIdxPEshort'].pct_change() * 100
    df_fii['propIdxCE_Long'] = df_fii['propIdxCElong'].pct_change() * 100

    "// Bearish "
    df_fii['fiiIdxFut_Short'] = df_fii['fiiIdxFutshort'].pct_change() * 100
    df_fii['fiiIdxCE_Short'] = df_fii['fiiIdxCEshort'].pct_change() * 100
    df_fii['fiiIdxPE_Long'] = df_fii['fiiIdxPElong'].pct_change() * 100
    df_fii['fiiStkFut_Short'] = df_fii['fiiStkFutshort'].pct_change() * 100
    df_fii['diiIdxFut_Short'] = df_fii['diiIdxFutshort'].pct_change() * 100
    df_fii['propIdxFut_Short'] = df_fii['propIdxFutshort'].pct_change() * 100
    df_fii['propIdxCE_Short'] = df_fii['propIdxCEshort'].pct_change() * 100
    df_fii['propIdxPE_Long'] = df_fii['propIdxPElong'].pct_change() * 100


    "// make new columns for gap, rank, ma_crossover and qty_sma"
    df_fii['gap_up'] = 100 * ((df_fii['open'] - df_fii['high'].shift(1)) / df_fii['high'].shift(1))
    df_fii['gap_down'] = 100 * ((df_fii['high'].shift(1) - df_fii['open']) / df_fii['high'].shift(1))
    df_fii['rank'] = 100 * ((df_fii['close'] - df_fii['low']) / (df_fii['high'] - df_fii['low']))
    #df_fii['rank_bearish'] = 100 * ((df_fii['low'] - df_fii['close']) / (df_fii['high'] - df_fii['low']))
    df_fii['rank_mean'] = df_fii['rank'].rolling(10).mean()
    # df_fii['rank_bearish_mean'] = df_fii['rank_bearish'].rolling(10).mean()

    df_fii['ma_crossover'] = 100 * (
                (df_fii['close'] - df_fii['close'].shift(1).rolling(20).mean()) / df_fii['close'].shift(1).rolling(
            20).mean())
    df_fii['qty_sma'] = 100 * (
                (df_fii['qty'] - df_fii['qty'].shift(1).rolling(10).mean()) / df_fii['qty'].shift(1).rolling(10).mean())
    df_fii = df_fii.drop(['date', 'symbol','adv','dec','new_high','new_low',
                          'trades',  'open', 'high', 'low', 'qty',
                          'diiIdxCEshort',
                          'diiIdxPEshort', 'diiIdxCElong', 'diiStkCEshort',
                          'diiStkCElong', 'diiStkPElong', 'diiStkPEshort', 'diiStkFutlong',
                          'diiStkFutshort', 'propStkPElong', 'propStkCElong', 'propStkCEshort',
                          'propStkPEshort', 'propStkFutshort', 'propStkFutlong', 'fiiIdxFutlong', 'fiiIdxFutshort',
                          'fiiStkCEshort', 'fiiStkPEshort', 'fiiStkPElong', 'fiiStkCElong',
                          'fiiIdxPEshort', 'fiiIdxCEshort', 'fiiIdxCElong', 'fiiIdxPElong',
                          'fiiStkFutlong', 'fiiStkFutshort', 'diiIdxFutlong', 'diiIdxFutshort',
                          'propIdxFutlong', 'propIdxFutshort', 'propIdxPEshort', 'propIdxCEshort',
                          'propIdxCElong', 'propIdxPElong'
                          ], axis=1)

    # col_re = df_fii.columns[-4:]
    # for cr in col_re:
    #     for i in range(1,num_days+1):
    #         df_fii[f'{cr}_lag_{i}'] = df_fii[cr].shift(i)
    # df_fii = df_fii.dropna()
    "// Drop OHLCV as oir objective is over"
    columns_to_change = df_fii.columns[:3]
    print(columns_to_change)
    "// Percentage change"
    df_fii[columns_to_change] = df_fii[columns_to_change].pct_change() * 100

    columns_for_mean = ['adv_dec_per', 'newH-newL_per', 'close',  'diiIdxPElong',
                        'fiiIdxFut_Long', 'fiiIdxPE_Short','avgprice',
                        'fiiIdxCE_Long', 'fiiStkFut_Long', 'diiIdxFut_Long',
                        'propIdxFut_Long', 'propIdxPE_Short', 'propIdxCE_Long',
                        'fiiIdxFut_Short', 'fiiIdxCE_Short', 'fiiIdxPE_Long',
                        'fiiStkFut_Short', 'diiIdxFut_Short',
                        'propIdxFut_Short', 'propIdxCE_Short', 'propIdxPE_Long'
                        ]
    for col in columns_for_mean:
        df_fii[f'{col}_mean'] = df_fii[col].rolling(10).mean()

    df_fii = df_fii.dropna()
    df_fii.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_fii.dropna(axis=1, how='any', inplace=True)

    if visualize:
        "// set target and visualize"
        df_fii['target'] = (df_fii['close'].shift(-1) > 0).astype(int)
        df_fii = df_fii.dropna()

        correlations = {}

        for col in df_fii.columns[:-1]:
            corr, pval = pointbiserialr(df_fii[col], df_fii['target'])
            correlations[col] = (corr, pval)

        dfcor = pd.DataFrame(correlations)
        dfcor = dfcor.transpose()
        dfcor.columns = ['Correlation', 'P-Value']
        dfcor['Combined'] = dfcor['Correlation'] * (1 - dfcor['P-Value'])  # Adjust weights as needed
        dfcor['Variable'] = dfcor.index
        dfcor = dfcor.reset_index(drop=True)
        dfcor = dfcor.sort_values(by='Variable')

        fig = px.bar(dfcor, x='Variable', y='Combined', color='Correlation',
                     text='Correlation', labels={'Combined': 'Combined Metric'},
                     title='Combined Metric (Correlation * (1 - P-Value))')

        # Customize layout
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(xaxis_title='Variable', yaxis_title='Combined Metric', coloraxis_showscale=False)
        plot(fig)

        "// get dataframe with highest +ve and -ve correlation"
        dfcor_plus = dfcor.sort_values(by='Combined', ascending=False)
        print(dfcor_plus['Variable'].to_list())
        dfcor_minus = dfcor.sort_values(by='Combined', ascending=True)
        print(dfcor_minus['Variable'].to_list())

    # highCor_columns = ['ad_delta_bullish', 'propIdxPE_Short', 'propIdxFut_Short_mean', 'propIdxPE_Long', 'rank_bullish',
    #                    'qty_sma_bullish', 'propIdxPE_Short_mean', 'propIdxCE_Long',
    #                    'ad_delta_bullish_mean', 'close',
    #                    'ma_crossover_bullish', 'fiiIdxPE_Short_mean', 'rank_bullish_mean', 'fiiIdxPE_Short',
    #                    'fiiIdxFut_Long_mean', 'avgprice', 'fiiIdxFut_Long', 'hl_delta_bullish', 'gap_up',
    #                    'ad_delta_bearish', 'rank_bearish', 'fiiIdxFut_Short_mean', 'fiiIdxFut_Short', 'qty_sma_bearish',
    #                    'ad_delta_bearish_mean', 'ma_crossover_bearish', 'rank_bearish_mean', 'propIdxFut_Long_mean',
    #                    'hl_delta_bearish', 'fiiIdxPE_Long', 'propIdxFut_Long', 'fiiIdxPE_Long_mean', 'gap_down']
    #
    # df_fii_highCorr = df_fii[highCor_columns]
    #

    highCor_columns2 = ['adv_dec_per','propIdxPE_Short', 'rank', 'diiIdxFut_Long_mean', 'close', 'ma_crossover',
                        'fiiIdxPE_Short', 'fiiIdxFut_Long_mean','gap_up',
                        'fiiIdxFut_Short_mean',  'fiiIdxPE_Long_mean', 'gap_down','fiiStkFut_Short_mean',
                        'diiIdxFut_Short', 'fiiStkFut_Short','newH-newL_per'
                        ]

    df_fii_highCorr2 = df_fii[highCor_columns2]

    "// using Random forest for extracting features "


    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    df_fii2 = df_fii.copy()
    df_fii2['target'] = (df_fii2['close'].shift(-1) > 0).astype(int)

    X = df_fii2.drop(['target'], axis=1)
    y = df_fii2['target']
    print(df_fii2.head(5))
    print(X.columns)
    # C Finally, let’s split the data into training and testing sets:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, shuffle=False)

    class_weights = {0: 0.65, 1: 0.50}
    rfdt_model = RandomForestClassifier(n_estimators=50, criterion='entropy', max_depth=15, min_samples_split=40,
                                        min_weight_fraction_leaf=0.07, random_state=99, class_weight=class_weights)
    rfdt_model = rfdt_model.fit(X_train, y_train)

    rfdt_pred = rfdt_model.predict(X_test)
    # Step 6: Evaluate the Model
    print(f"Classification Report:")
    print(classification_report(y_test, rfdt_pred))

    "// feature extraction"
    feature_importances = rfdt_model.feature_importances_

    feature_names = X_train.columns

    # Create a DataFrame with feature names and their importances
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

    # Sort the DataFrame by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    feature_list = feature_importance_df.head(8).Feature.to_list()
    df_fii_rf = df_fii2[feature_list]

    return df_fii,  df_fii_highCorr2, df_fii_rf
