import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree,DecisionTreeRegressor, export_graphviz
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pydot
import graphviz
from IPython.display import Image
import seaborn as sns

root = 'd:/demos/'
df = pd.read_csv(root+'nifty_index_data_hist_old.csv')

# c = df.iloc[-1]['close']

df['high'] = df['high'].pct_change(1)
df['high'] = df['high'].multiply(100)
df['low'] = df['low'].pct_change(1)
df['low'] = df['low'].multiply(100)
df['qty'] = df['qty'].pct_change(1)
df['qty'] = df['qty'].multiply(100)
df['trades'] = df['trades'].pct_change(1)
df['trades'] = df['trades'].multiply(100)
df['gap'] = df['open'] - df['close'].shift(1)
df['gap_per'] = 100*(df['gap']/df['close'].shift(1))

df['close'] = df['close'].pct_change(1)
df['return'] = df['close'].multiply(100)
df['ad_ratio'] = df['adv']/df['dec']
df['hl_ratio'] = df['new_high']/df['new_low']

df = df.dropna()

#df['Target'] = df.copy()['close'] <= df.copy()['close'].shift(-1)
df['Target'] = np.where(df['close'].shift(-1) >= df['close'], 1, -1)


# df['Target'] = df['Target1'].shift(-1)
# print(df[['close','Target','Target1']])

# df = df.dropna()
df = df.drop(['date', 'open', 'symbol', 'adv', 'dec', 'new_high', 'new_low','close','gap'], axis=1)
# df = df   .drop(['Q%','date','close','dayopen','valueavg','VAH','VAL','DailyRange'],axis=1)
# df = df.drop(['date','dayopen','POC','DailyRange','high','low'],axis=1)
# df2=df.copy()
# print(df2)
features = list(df.columns[:8])
# features = list(df.columns[:13])
print(features)
print(df.head())

# Use your brain as the greatest neural network using this following line and analyze the chart
# sns.set(style='dark')
sns.pairplot(hue='Target', palette='Set1', data=df)  # To plot more visual type chart, no targets shown
# sns.pairplot( data=df.tail(100))  # to plot along with target, useful for checking If data has equal number of samples

from sklearn.model_selection import train_test_split

y = df["Target"]
x = df[features]

# While splitting the data between training and test data shuffle is set to true by default but
# since we are dealing with time series data it's better to set to false.
# Training data will have first 70% of the data and test data will have last 30% of the data in deterministic manner

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, shuffle=False)


# print(df)
# print(list(df2.Target))

# dt = DecisionTreeRegressor() #criterion = 'gini'
"""
Gini impurity and entropy are criteria used in the construction of decision trees, 
particularly in the context of the 
CART (Classification and Regression Trees) algorithm. These criteria help decide 
how to split the data at each node of the tree.

Entropy: It measures the average information content of a set.
 In the context of decision trees, entropy is a measure of impurity or disorder
"""
dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=40, random_state=99,
                                                        max_depth=5, min_weight_fraction_leaf=0.05)

dt = dt.fit(x_train, y_train)

class_names = [str(class_label) for class_label in set(y_train)]


"""
important:
 download graphviz.exe
 https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/9.0.0/windows_10_cmake_Release_graphviz-install-9.0.0-win64.exe
 install while selecting option for add system user path 
 go to installation folder of c:\programfiles\graphviz\bin and copy dot.exe and paste in anaconda main folder
 assuming you have already installed Python with Anaconda if not put dot.exe in the folder where python.exe exist 
"""

dot_data = tree.export_graphviz(dt, out_file=None, feature_names=features, class_names=class_names, filled=True, node_ids=True,
                     proportion=False)



graph = graphviz.Source(dot_data, format="png")
graph.render("d:/demos/decision_tree3")

# Display the decision tree graph using Plotly
image_path = "d:/demos/decision_tree3.png"
Image(filename=image_path)

# (graph,) = pydot.graph_from_dot_file('treei.dot')
# graph.write_png('d:/demos/MLtreei.png')

# ..........................................................

y_pred = dt.predict(x_test)
acc = accuracy_score(y_test, y_pred) * 100
print(acc)

# dftest = df.drop(['Target'], axis=1)
# # testf = dftest.tail(1).values
#
# # ........................................................................................
# testf = dftest.iloc[-1].values
# # ..........................................................................................
# testf = testf.reshape(1, -1)
#
# print('rows are')
# print(testf)
#
# pred = dt.predict_proba(testf, check_input=True)
# predbi = dt.predict(testf)
#
# path = dt.decision_path(testf, check_input=True)
# print(path)
#
# node_indicator = dt.decision_path(testf, check_input=True)
# sample_id = 0
#
# node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
#                                     node_indicator.indptr[sample_id + 1]]
# print(node_index)
# # ..............................................................................
#
#
# n_nodes = dt.apply(testf)
# leave_id = dt.apply(testf)
# print(n_nodes)
# # n_nodes = dt.tree_.node_count
# children_left = dt.tree_.children_left
# children_right = dt.tree_.children_right
# feature = dt.tree_.feature
# threshold = dt.tree_.threshold
# treeval = dt.tree_.value
# pathsize = len(node_index)
# # print('pathsize', pathsize)
#
#
# rulelist = []
#
# for ind in range(pathsize):
#
#     v1 = (node_index.item(ind))  # node number
#     v2 = feature.item(v1)  # feature number by column
#     lastf = testf.item(v2)
#     lastf = round(lastf, 2)
#     # print('feature VAL ', v2)
#
#     if v2 >= 0:
#         print(v1)
#         if (testf.item(v2) <= threshold.item(v1)):
#             sign = " <= "
#         else:
#             sign = " > "
#
#         rule1 = ('If ' + str(features[v2]) + sign + str(round((threshold.item(v1)), 2)) + '( last value ' + str(
#             lastf) + ' )')
#         if ind < pathsize - 2:
#             rule = rule1 + ' and '
#         else:
#             rule = rule1
#         rulelist.append(rule)

# test = treeval[n_nodes]

# print('Rules' + '\n' + str(rulelist) + '\n' + 'Number of Bearish & Bullish days ' + str(test) + ' respectively')
# ........................    ............................................................................


# impdf = pd.DataFrame(dt.feature_importances_,columns=['Imp'],index=X.columns).sort(['Imp'],ascending=False) #feature importance score

impdf = pd.DataFrame(dt.feature_importances_, columns=['Imp'], index=x.columns).sort_values(by='Imp',
                                                                                            ascending=False)  # feature importance score
print(impdf)


# .................................. modification random forest


rfdt = RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=7, min_samples_split=40,
                             min_weight_fraction_leaf=0.05,  random_state=99)
rfdt = rfdt.fit(x, y)
y_pred = rfdt.predict(x_test)
acc2 = accuracy_score(y_test, y_pred) * 100
print(acc2)
