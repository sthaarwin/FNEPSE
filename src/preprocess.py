import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pathlib import Path

def preprocess_patterns(csv_path="data/raw/merged-stock-data.csv", lookback = 10):
    #load the dataset
    df = pd.read_csv(csv_path)
    
    print(f"Loaded dataset with {len(df)} rows")
    print(f"Unique patterns: {df['pattern'].nunique()}")
    print(f"Pattern distribution:\n{df['pattern'].value_counts()}")

    #features OHLCV
    feature_cols = ["open", "high", "low", "close"]
    X_raw = df[feature_cols].values

    #label encoding
    le = LabelEncoder()
    Y_raw = le.fit_transform(df["pattern"].values)

    #scale the features to 0-1
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    #converting to windows
    x, y = [], []

    for i in range(lookback, len(X_scaled)):
        x.append(X_scaled[i-lookback:i])
        y.append(Y_raw[i])
        
    x, y = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

    #train/val/test split 70/15/15
    n       = len(x)
    n_train = int(n*0.7)
    n_val   = int(n*0.15)
    n_test  = int(n*0.15)

    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val     = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
    x_test, y_test   = x[n_train + n_val:], y[n_train + n_val:]

    #save
    Path("data/preprocessed").mkdir(parents=True, exist_ok=True)
    np.savez("data/preprocessed/stock_dataset.npz",
             x_train = x_train, y_train = y_train,
             x_val = x_val, y_val = y_val,
             x_test = x_test, y_test = y_test,
             classes = le.classes_
             )

    print("Processing completed")
    print("Classes : ", le.classes_)
    print("Shapes  : ", x_train.shape, x_val.shape, x_test.shape)

    return le.classes_

if __name__ == "__main__":
    preprocess_patterns()