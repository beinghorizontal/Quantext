import pandas as pd
import plotly.express as px
from plotly.offline import plot
from scipy.stats import pointbiserialr

# C, for historical data
def get_options():

    df_options_raw = pd.read_csv('optionchain_data.csv')
    # print(df_options_raw.columns)
    df_options_raw = df_options_raw.set_index(df_options_raw['TimeIndex'], drop=True)
    column_mapping = {'CEIV_Mean': 'CEIV', 'PEIV_Mean': 'PEIV'}
    df_options_raw.rename(columns=column_mapping, inplace=True)

    selected_columns = ['MaxPain', 'ActivePE', 'ActiveCE', 'MX_PE', 'MX_CE']

    "// Calculate the difference between close and selected columns"
    df_diff_selected = df_options_raw[selected_columns].subtract(df_options_raw['Nifty_Close'], axis=0)
    "// transform to %"
    df_diff_selected['MaxPain'] = 100 * (df_diff_selected['MaxPain'] / df_options_raw['Nifty_Close'])
    df_diff_selected['ActivePE'] = 100 * (df_diff_selected['ActivePE'] / df_options_raw['Nifty_Close'])
    df_diff_selected['ActiveCE'] = 100 * (df_diff_selected['ActiveCE'] / df_options_raw['Nifty_Close'])
    df_diff_selected['MX_PE'] = 100 * (df_diff_selected['MX_PE'] / df_options_raw['Nifty_Close'])
    df_diff_selected['MX_CE'] = 100 * (df_diff_selected['MX_CE'] / df_options_raw['Nifty_Close'])

    # Display the resulting DataFrame
    # print(df_diff_selected)

    df_drop = df_options_raw.drop(['TimeIndex', 'Nifty_O', 'Nifty_H', 'Nifty_L', 'MaxPain', 'ActivePE',
                                   'ActiveCE', 'MX_PE', 'MX_CE'
                                   ], axis=1)
    df_options = pd.concat([df_drop, df_diff_selected], axis=1)

    df_options = df_options.apply(pd.to_numeric, errors='ignore')

    df_options = df_options.replace(0, 0.001)

    "// Calculate IV rank and IV percentile"
    df_options['peIvRank'] = 100 * (
            (df_options['PEIV'] - df_options['PEIV'].rolling(10).min()) /
            (df_options['PEIV'].rolling(10).max() - df_options['PEIV'].rolling(10).min()))
    df_options['ceIvRank'] = 100 * (
            (df_options['CEIV'] - df_options['CEIV'].rolling(10).min()) /
            (df_options['CEIV'].rolling(10).max() - df_options['CEIV'].rolling(10).min()))

    df_options = df_options.dropna()

    # df_options['bullishOi'] = df_options['Long_Sum'] + df_options['ShortCover_Sum']
    # df_options['bearishOi'] = df_options['Short_Sum'] + df_options['LongLiq_Sum']
    # # df_options['bearishOi2'] = df_options['LongLiq_Sum'] + df_options['LongLiq_Sum']
    #
    # # df_options['netBuildUp'] = df_options['bullishOi'] - df_options['bearishOi']
    # df_options['netBullishBuildUp'] = df_options['bullishOi'] - df_options['bearishOi']
    # df_options['netBearishBuildUp'] = df_options['bearishOi'] - df_options['bullishOi']
    # df_options['Option_BullishUnwind'] = df_options['CE_Unwind_Sum'] - df_options['PE_Unwind_Sum']
    # df_options['Option_BearishhUnwind'] = df_options['PE_Unwind_Sum'] - df_options['CE_Unwind_Sum']
    # df_options['netBuildUp3'] = (df_options['netBuildUp2'] - df_options['Option_BullishUnwind'])- df_options['bearishOi']

    # print(custom_tabulate(df[['avgprice','target']]))
    # df_options.columns

    columns_for_pct = ['MX_PEOI', 'MX_CEOI', 'Nifty_Close', 'Nifty_V', 'CEIV', 'PEIV', 'PCR_OI',
                       'PCR_WT', 'ActivePEOI', 'ActiveCEOI']

    df_options[columns_for_pct] = df_options[columns_for_pct].pct_change() * 100
    df_options = df_options.dropna()
    # print((df_options.columns.tolist()))

    cols_for_mean = ['MX_PEOI', 'MX_CEOI', 'Nifty_Close', 'Nifty_V', 'CEIV', 'PEIV', 'PCR_OI', 'PCR_WT',
                     'ActivePEOI', 'ActiveCEOI', 'Long_Sum', 'Short_Sum', 'LongLiq_Sum', 'ShortCover_Sum', 'CE_Unwind_Sum',
                     'PE_Unwind_Sum', 'MaxPain', 'ActivePE', 'ActiveCE', 'MX_PE', 'MX_CE', 'peIvRank', 'ceIvRank'
                      ]

    for col in cols_for_mean:
        df_options[f'{col}_mean'] = df_options[f'{col}'].rolling(10).mean()

    df_options = df_options.dropna()
    visualize = False
    if visualize:

        df_options['target'] = (df_options['Nifty_Close'].shift(-1) > 0).astype(int)
        # continuous_columns = df_options.iloc[:, 0:28]
        # binary_column = df_options['target']

        correlations = {}
        for col in df_options.columns[:-1]:
            corr, pval = pointbiserialr(df_options[col], df_options['target'])
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

        dfcor_plus = dfcor.sort_values(by='Combined', ascending=False)
        print(dfcor_plus['Variable'].to_list())
        dfcor_minus = dfcor.sort_values(by='Combined', ascending=True)
        print(dfcor_minus['Variable'].to_list())

    # highCor_columns = ['Option_BullishUnwind', 'Nifty_Close', 'Option_BullishUnwind_mean', 'bullishOi', 'netBullishBuildUp',
    #                    'ShortCover_Sum_mean', 'CE_Unwind_Sum', 'Nifty_Close_mean', 'LongLiq_Sum_mean', 'ShortCover_Sum',
    #                    'PCR_OI', 'bullishOi_mean', 'netBullishBuildUp_mean', 'PCR_OI_mean', 'Long_Sum', 'ActiveCE',
    #                    'ActiveCE_mean', 'PEIV', 'MX_PE', 'ceIvRank_mean', 'MX_CEOI_mean', 'MX_PE_mean',
    #                    'peIvRank_mean', 'Short_Sum_mean', 'ActiveCEOI', 'Option_BearishhUnwind', 'MX_CEOI',
    #                    'PE_Unwind_Sum_mean', 'Option_BearishhUnwind_mean', 'LongLiq_Sum', 'MaxPain', 'netBearishBuildUp',
    #                    'bearishOi', 'PE_Unwind_Sum', 'MX_PEOI', 'ceIvRank', 'ActivePEOI', 'netBearishBuildUp_mean',
    #                    'bearishOi_mean', 'ActiveCEOI_mean', 'MX_CE_mean', 'Nifty_V', 'MaxPain_mean', 'CEIV_mean',
    #                    'ActivePEOI_mean', 'PEIV_mean', 'MX_PEOI_mean', 'Long_Sum_mean']
    #
    # df_options_high_corr = df_options[highCor_columns]

    highCor_columns2 = [
        'Nifty_Close', 'ShortCover_Sum_mean', 'CE_Unwind_Sum', 'PCR_OI','ShortCover_Sum',
        'Long_Sum', 'ActiveCE',
        'Short_Sum_mean', 'ActiveCEOI', 'MX_CEOI', 'PE_Unwind_Sum_mean', 'LongLiq_Sum','MaxPain',
        'PE_Unwind_Sum','MX_PEOI',
        'ceIvRank', 'ActiveCEOI_mean', 'ActivePEOI', 'MX_CE_mean'
    ]

    df_options_high_corr2 = df_options[highCor_columns2]

    "// using Random forest for extracting features "

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    df_options2 = df_options.copy()
    df_options2['target'] = (df_options2['Nifty_Close'].shift(-1) > 0).astype(int)

    X = df_options2.drop(['target'], axis=1)
    y = df_options2['target']
    print(df_options2.head(5))
    print(X.columns)
    # C Finally, let’s split the data into training and testing sets:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, shuffle=False)

    class_weights = {0: 0.54, 1: 0.50}
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
    df_options_rf = df_options[feature_list]
    return df_options,  df_options_high_corr2, df_options_rf
