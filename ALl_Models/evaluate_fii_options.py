import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from option_vis_corr_2 import get_options
from fii_vis_corr_2 import get_fii
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
"// concat"
df_fii_list = get_fii()
df_option_list = get_options()

df_fii = df_fii_list[1]  # List 0:raw DF (all features), list1:manual feature DF, list2:RF generated feature DF
df_options = df_option_list[1]

df_merge = pd.concat([df_fii, df_options], axis=1)
df_merge = df_merge.dropna()

"//set target"
df_merge['target'] = (df_merge['close'].shift(-1) > 0).astype(int)
df_merge = df_merge.dropna()
df_merge = df_merge.drop(['Nifty_Close'],axis=1)
X = df_merge.drop(['target'], axis=1)
y = df_merge['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, shuffle=False)
# C And normalize our features:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test[np.isinf(X)] = 0
X_test_scaled = scaler.fit_transform(X_test)

# C visualize scale difference
df_scaled = pd.DataFrame(X_train_scaled)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.50, 0.50],vertical_spacing=0.003,horizontal_spacing=0.0003,
                    specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=X_train.index, y=df_scaled[0],mode='lines+markers',
                         name='<span style="color:green">Scaled Data </span>',marker=dict(color='green')),secondary_y=False,row=1,col=1)


fig.add_trace(go.Scatter(x=X_train.index, y=X_train['propIdxCE_Long_mean'], mode='lines+markers',
                         name='<span style="color:red">UnScaled Data</span>',marker=dict(color='red')),
              secondary_y=False,row=2,col=1)
fig.update_xaxes(showline=False, color='white', showgrid=False, type='category',
                    tickangle=90, zeroline=False, row=2)

fig.update_yaxes(showline=False, color='white', showgrid=False,
                    row=2)
fig.update_yaxes(showline=False, color='white', showgrid=False,
                     row=1)

fig.update_layout(paper_bgcolor='black',plot_bgcolor='black',
                  height=500, width=1200,autosize=False,
                  xaxis =dict(showgrid=False,rangeslider=dict(visible=False)),
                  yaxis =dict(showgrid=False, tickformat='d'),
                  yaxis2=dict(showgrid=False, tickformat='d'))
plot(fig, auto_open=True)


# C ................................................. Linear regression and Decision Tree then compare with SVM.....
log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train_scaled, y_train)

# Step 5: Make Predictions
log_pred = log_model.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, log_pred))


"// Decision Tree simple"
dt_model = DecisionTreeClassifier(criterion='entropy', min_samples_split=40, random_state=99,
                                                        max_depth=15, min_weight_fraction_leaf=0.07)

dt_model = dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("Classification Report Decision Tree:")
print(classification_report(y_test, dt_pred))

"// Random Forest"

rfdt_model = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=15, min_samples_split=40,
                                    min_weight_fraction_leaf=0.07, random_state=99)
rfdt_model = rfdt_model.fit(X_train, y_train)

rfdt_pred = rfdt_model.predict(X_test)
# Step 6: Evaluate the Model
print(f"Classification Report:")
print(classification_report(y_test, rfdt_pred))

"// Adjust the weights to improve recall for class 0"
class_weights = {0: 0.58, 1: 0.50}
rfdt_model = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=15, min_samples_split=40,
                                    min_weight_fraction_leaf=0.07, random_state=99, class_weight=class_weights)
rfdt_model = rfdt_model.fit(X_train, y_train)

rfdt_pred = rfdt_model.predict(X_test)
# Step 6: Evaluate the Model
print(f"Classification Report:")
print(classification_report(y_test, rfdt_pred))

" // SVM"

best_params = {'C': 2, 'gamma': 0.05, 'kernel': 'rbf'}
class_weights_svm = {0: 0.50, 1: 0.50}
final_svm_model = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'], probability=True,
                      class_weight=class_weights_svm)

final_svm_model.fit(X_train_scaled, y_train)

final_svm_model_pred = final_svm_model.predict(X_test_scaled)
print(f"Classification Report SVM:")
print(classification_report(y_test, final_svm_model_pred))

# C probability method won't work here because SVM use decision boundaries and sample distance from those boundaries
final_svm_probabilities = final_svm_model.predict_proba(X_test_scaled)
df_final_svm_probabilities = pd.DataFrame(final_svm_probabilities)
# C Binary class prediction values won't match here
df_final_svm_probabilities['predictions'] = final_svm_model_pred

# C Correct function. It uses decision boundary. Binary class prediction values will match
final_svm_boundry = final_svm_model.decision_function(X_test_scaled)
df_final_svm_boundry = pd.DataFrame(final_svm_boundry)
df_final_svm_boundry['positive_pred'] = np.where(df_final_svm_boundry[0]>0,df_final_svm_boundry[0],0)
df_final_svm_boundry['negative_pred'] = np.where(df_final_svm_boundry[0]<0,df_final_svm_boundry[0],0)
df_final_svm_boundry['binary_prediction'] = final_svm_model_pred
print(df_final_svm_boundry.head(2).to_dict())
# C visualize decision boundary
print(df_final_svm_boundry.head(5)[['positive_pred','negative_pred']].to_dict())

# Sort the DataFrame by the decision function values for better visualization
df_sorted = df_final_svm_boundry.copy()
df_sorted['x'] = df_sorted[0].abs()
df_sorted = df_sorted.sort_values('x',ascending=True)

fig = go.Figure()

# Scatter plot for positive predictions
fig.add_trace(go.Scatter(
    x=abs(df_sorted['x']),
    y=df_final_svm_boundry['positive_pred'],
    mode='markers',
    marker=dict(color='green', symbol='circle'),
    name='Positive Predictions'
))

# Scatter plot for negative predictions
fig.add_trace(go.Scatter(
    x=df_sorted['x'],
    y=df_final_svm_boundry['negative_pred'],
    mode='markers',
    marker=dict(color='red', symbol='circle'),
    name='Negative Predictions'
))

# Decision line
fig.add_trace(go.Scatter(
    x=df_sorted['x'],
    y=[0] * len(df_sorted),
    mode='lines',
    line=dict(color='blue', width=2),
    name='Decision Line'
))

# Layout customization
fig.update_layout(
    title='Sample Distribution and Decision Line',
    xaxis_title='Index Number',
    yaxis_title='Prediction Values',
)

# Show the plot
plot(fig)


"// Backtest"
dfBacktest = df_final_svm_boundry.copy()
dfBacktest['date'] = X_test.index
dfBacktest = dfBacktest.rename(columns={0: 'distFromBoundary'})
dfBacktest = dfBacktest.set_index(dfBacktest['date'], drop=True)
dfBacktest['actual_outcome'] = y_test
dfBacktest = dfBacktest.drop(['date'],axis=1)
threshold = 1
winner_long = ((dfBacktest['positive_pred'] > threshold) & (dfBacktest['actual_outcome'] == 1)).sum() / (
            dfBacktest['positive_pred'] >= threshold).sum()

winner_short = ((dfBacktest['negative_pred'] < -threshold) & (dfBacktest['actual_outcome'] == 0)).sum() / (
            dfBacktest['negative_pred'] < -threshold).sum()

print(f"winners lomg: {round(winner_long,2)}% Out of: {(dfBacktest['positive_pred'] >= 0.5).sum()}"
      f"\nwinners short: {round(winner_short,2)}% Out of: {(dfBacktest['negative_pred'] < -0.5).sum()}")

from supervised.automl import AutoML

automl = AutoML(mode='Explain')
automl.fit(X_train, y_train)

y_pred = automl.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
y_pred.view()
print(f"Accuracy: {accuracy_score(X_test['propIdxCE_Long_mean'], y_pred)*100.0:.2f}%" )



"// xgboost "
from xgboost import XGBClassifier

"//Calculate the ratio of class 0 to class 1 samples"
num_class_0 = len(y_train) - sum(y_train)
num_class_1 = sum(y_train)
scale_pos_weight = num_class_0 / num_class_1
model = XGBClassifier(
    learning_rate=0.01,
    n_estimators=1000,
    max_depth=15,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    objective='binary:logistic',
    eval_metric='error', scale_pos_weight=scale_pos_weight
)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

print(f"Classification Report SVM:")
print(classification_report(y_test, final_svm_model_pred))

