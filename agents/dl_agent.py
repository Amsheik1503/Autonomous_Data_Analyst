# agents/dl_agent.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def create_sequences(series, n_lags):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i - n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def dl_agent(df, target_col, n_lags=7, epochs=50, batch_size=32, test_size=0.2):
    print("\n🧠 DL Agent: Training deep learning models...")

    series = df[target_col].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series)

    # -----------------------------------
    # Clean & prepare series
    # -----------------------------------
    series_df = df[[target_col]].dropna().reset_index(drop=True)
    series = series_df[target_col].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series)

    # -----------------------------------
    # Train-test split (time-aware)
    # -----------------------------------
    split_idx = int(len(series_scaled) * 0.8)
    train_series = series_scaled[:split_idx]
    test_series = series_scaled[split_idx:]

    # -----------------------------------
    # Create sequences
    # -----------------------------------
    X_train, y_train = create_sequences(train_series, n_lags)
    X_test, y_test = create_sequences(test_series, n_lags)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    results = {}
    models = {}
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)

    # ----------------- LSTM -----------------
    lstm = Sequential([LSTM(50, input_shape=(n_lags, 1)), Dense(1)])
    lstm.compile(optimizer="adam", loss="mse")
    lstm.fit(X_train, y_train, validation_split=0.1, epochs=epochs,
             batch_size=batch_size, verbose=0, callbacks=[early_stop])
    preds = lstm.predict(X_test, verbose=0)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = np.mean(np.abs(y_test - preds))
    r2 = 1 - np.sum((y_test - preds) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    results["LSTM"] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    models["LSTM"] = lstm

    # ----------------- BiLSTM -----------------
    bilstm = Sequential([Bidirectional(LSTM(50), input_shape=(n_lags,1)), Dense(1)])
    bilstm.compile(optimizer="adam", loss="mse")
    bilstm.fit(X_train, y_train, validation_split=0.1, epochs=epochs,
               batch_size=batch_size, verbose=0, callbacks=[early_stop])
    preds = bilstm.predict(X_test, verbose=0)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = np.mean(np.abs(y_test - preds))
    r2 = 1 - np.sum((y_test - preds) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    results["BiLSTM"] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    models["BiLSTM"] = bilstm

    # ----------------- GRU -----------------
    gru = Sequential([GRU(50, input_shape=(n_lags,1)), Dense(1)])
    gru.compile(optimizer="adam", loss="mse")
    gru.fit(X_train, y_train, validation_split=0.1, epochs=epochs,
            batch_size=batch_size, verbose=0, callbacks=[early_stop])
    preds = gru.predict(X_test, verbose=0)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = np.mean(np.abs(y_test - preds))
    r2 = 1 - np.sum((y_test - preds) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    results["GRU"] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    models["GRU"] = gru

    # ----------------- CNN-LSTM -----------------
    cnn_lstm = Sequential([Conv1D(64,2,activation="relu",input_shape=(n_lags,1)),
                           MaxPooling1D(pool_size=2),
                           LSTM(50),
                           Dense(1)])
    cnn_lstm.compile(optimizer="adam", loss="mse")
    cnn_lstm.fit(X_train, y_train, validation_split=0.1, epochs=epochs,
                 batch_size=batch_size, verbose=0, callbacks=[early_stop])
    preds = cnn_lstm.predict(X_test, verbose=0)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = np.mean(np.abs(y_test - preds))
    r2 = 1 - np.sum((y_test - preds) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    results["CNN-LSTM"] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    models["CNN-LSTM"] = cnn_lstm

    results_df = pd.DataFrame(results).T.sort_values("RMSE")
    best_model_name = results_df.index[0]
    best_model = models[best_model_name]

    print(f"\n🏆 Best DL Model: {best_model_name}")
    return best_model_name, best_model, results_df
