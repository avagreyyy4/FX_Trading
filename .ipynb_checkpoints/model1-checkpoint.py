# IMPORTING LIBRARIES 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier 
import sklearn.datasets
import sklearn.preprocessing
import sklearn.random_projection
import sklearn.neighbors
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt


# define patterns
n = 5
profit_taking = 0.0025
p = 0.70

#READ IN DATA
df_original = pd.read_csv("data/final_clean_FX_data.csv")

#feature engineer
df_original = df_original.sort_values(by =  "Date", ascending=True)
df = df_original
df["c_over_o"] = (df["Close"] - df["Open"]) / df["Open"]
df["h_over_o"] = (df["High"]  - df["Open"]) / df["Open"]
df["l_over_o"] = (df["Low"]   - df["Open"]) / df["Open"]
df["range"] = (df["High"] - df["Low"]) / df["Open"]


# CREATE PRIOR n DAYS FEATURE
for before in range(1, n+1):
    df[f"Close_{before}_before"] = df["Close"].shift(before)
    df[f"Open_{before}_before"] = df["Open"].shift(before)
    df[f"High_{before}_before"] = df["High"].shift(before)
    df[f"Low_{before}_before"] = df["Low"].shift(before)
    df[f"c_over_o_lag{before}"] = df["c_over_o"].shift(before)
    df[f"h_over_o_lag{before}"] = df["h_over_o"].shift(before)
    df[f"l_over_o_lag{before}"] = df["l_over_o"].shift(before)

#feature enginnering
df["range_lag1"] = df["range"].shift(1)
df["range_5d"] = df["range_lag1"].rolling(5).mean()


df["ret_1d"] = np.log(df["Close"]) - np.log(df["Close_1_before"])
df["ret_5d"] = np.log(df["Close"]) - np.log(df["Close_5_before"])
df["vol_5d"]  = df["ret_1d"].rolling(5).std()
df["vol_10d"] = df["ret_1d"].rolling(10).std()
df["mom_5d"] = df["ret_1d"].rolling(5).mean()
df["mom_10d"] = df["ret_1d"].rolling(10).mean()


feature_cols = [
    "ret_1d", "ret_5d",
    "vol_5d", "vol_10d",
    "mom_5d", "mom_10d", 
    "range_5d"
]

df[feature_cols] = df[feature_cols].shift(1)

df["target"] = (df["High"] > (profit_taking+1)*df["Open"]).astype(int)
target = df[["target"]]
df = df.dropna()
df = df.drop(columns=['c_over_o', 'h_over_o', 'l_over_o', 'range'])



train_df   = df[(df['Date'] >= '2017-01-01') & (df['Date'] < '2020-01-01')]
val_df   = df[(df['Date'] >= '2020-01-01') & (df['Date'] < '2022-01-01')]
test_df  = df[df['Date'] >= '2022-01-01']

x_train = train_df.drop(columns=['target', 'Date', 'Close', 'Open', 'High', 'Low'])
y_train = train_df['target']

x_val = val_df.drop(columns=['target', 'Date', 'Close', 'Open', 'High', 'Low'])
y_val = val_df['target']

x_test = test_df.drop(columns=['target', 'Date', 'Close', 'Open', 'High', 'Low'])
y_test = test_df['target']

# Scaling the data

standardize = StandardScaler(with_mean=True, with_std=True)
standardize.fit(x_train)
x_train = pd.DataFrame(
    standardize.transform(x_train),
    columns=x_train.columns,
    index=x_train.index
)
x_val = pd.DataFrame(
    standardize.transform(x_val),
    columns=x_val.columns,
    index=x_val.index
)
x_test = pd.DataFrame(
    standardize.transform(x_test),
    columns=x_test.columns,
    index=x_test.index
)



model1 = MLPClassifier(
    hidden_layer_sizes=[16, 16],
    activation="relu",
    alpha=0.001,
    max_iter=1000,
    random_state = 42
)

model1.fit(x_train, y_train)

validation_accuracy = model1.score(x_val, y_val)
print(f"validation_accuracy1={validation_accuracy:0.4f}")
train_accuracy = model1.score(x_train, y_train)
print(f"train_accuracy1={train_accuracy:0.4f}")

if True:
    test_accuracy = model1.score(x_test, y_test)
    print(f"test_accuracy={test_accuracy}")

    y_true = y_test.values.ravel()
    probs_test = model1.predict_proba(x_test)[:, 1]
    preds_test = model1.predict(x_test)


#eval NN

## cleaning data
X_test = test_df.drop(columns=["target", "Date", "Close", "Open", "High", "Low"])
y_test = test_df["target"]
X_test_scaled = standardize.transform(X_test)

## probabilities
probs = model1.predict_proba(X_test_scaled)[:, 1]

# probability distribution
print("Probability summary:")
print(pd.Series(probs).describe())

# ROC-accuracy
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, probs)
print(f"ROC-AUC: {auc:.3f}")


# Quintile hit-rate 
df_eval = test_df.copy()
df_eval["prob"] = probs

df_eval["quintile"] = pd.qcut(df_eval["prob"], 5, labels=False)

quintile_perf = (
    df_eval
    .groupby("quintile")["target"]
    .mean()
)

print("TP hit-rate by probability quintile:")
print(quintile_perf)

# threshold-based trading behavior
df_eval["enter"] = (df_eval["prob"] >= p).astype(int)

num_trades = df_eval["enter"].sum()
trade_hit_rate = df_eval.loc[df_eval["enter"] == 1, "target"].mean()
base_hit_rate = df_eval["target"].mean()

print(f"Base hit rate: {base_hit_rate:.3f}")
print(f"Trades taken: {num_trades}")
print(f"Hit rate on trades: {trade_hit_rate:.3f}")


precision = precision_score(
    y_test,
    df_eval["enter"]
)

print(f"Precision (hit rate on entered days): {precision:.3f}")

coverage = df_eval["enter"].mean()

print(f"Coverage (fraction of days entered): {coverage:.3f}")

## model peformance
df_pnl = df_eval.copy()

# Open → Close return
df_pnl["ret_oc"] = (df_pnl["Close"] - df_pnl["Open"]) / df_pnl["Open"]

df_pnl["daily_return"] = 0.0

# If we enter and TP hits
df_pnl.loc[
    (df_pnl["enter"] == 1) & (df_pnl["target"] == 1),
    "daily_return"
] = profit_taking

# If we enter and TP does NOT hit
df_pnl.loc[
    (df_pnl["enter"] == 1) & (df_pnl["target"] == 0),
    "daily_return"
] = df_pnl["ret_oc"]

# Compound equity
df_pnl["equity_model"] = (1 + df_pnl["daily_return"]).cumprod()

model_final_equity = df_pnl["equity_model"].iloc[-1]
print(f"Final equity (model strategy): {model_final_equity:.3f}")


#baseline performance

df_base = test_df.copy()

df_base["ret_oc"] = (df_base["Close"] - df_base["Open"]) / df_base["Open"]

df_base["daily_return"] = df_base["ret_oc"]

# Override with TP when hit
df_base.loc[
    df_base["target"] == 1,
    "daily_return"
] = profit_taking

df_base["equity_base"] = (1 + df_base["daily_return"]).cumprod()

base_final_equity = df_base["equity_base"].iloc[-1]
print(f"Final equity (enter every day): {base_final_equity:.3f}")

relative_performance = model_final_equity / base_final_equity
print(f"Relative performance (model / baseline): {relative_performance:.3f}x")

