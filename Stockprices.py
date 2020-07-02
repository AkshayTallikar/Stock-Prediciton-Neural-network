import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("AAPL (2).csv", date_parser=True)
print(data.head())

data_training = data[data['Date'] < '2019-01-02'].copy()
data_test = data[data['Date'] > '2019-01-02'].copy()

training_data = data_training.drop(['Date', 'Adj Close'], axis=1)
training_test = data_test.drop(['Date', 'Adj Close'], axis=1)

scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)
print(training_data)
X_train = []
y_train = []

for i in range(60, training_data.shape[0]):
    X_train.append((training_data[i - 60:i]))
    y_train.append((training_data[i, 0]))

X_train, y_train = np.array(X_train), np.array(y_train)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

regression = Sequential()
regression.add(LSTM(units=60, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 5)))
regression.add(Dropout(.2))
regression.add(LSTM(units=60, activation='relu', return_sequences=True, ))
regression.add(Dropout(.2))
regression.add(LSTM(units=80, activation='relu', return_sequences=True, ))
regression.add(Dropout(.2))
regression.add(LSTM(units=120, activation='relu'))
regression.add(Dropout(.2))
regression.add(Dense(units=12))
regression.summary()
regression.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])
regression.fit(X_train, y_train, epochs=9, batch_size=32)

past_60_days = data_training.tail(60)
df = past_60_days.append(data_test, ignore_index=True)
df = df.drop(['Date', 'Adj Close'], axis=1)
inputs = scaler.transform(df)
X_test = []
y_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i - 60:i])
    y_test.append(inputs[i, 0])

X_test = np.array(X_test)
y_test = np.array(y_test)
y_pred = regression.predict(X_test)

plt.figure(figsize=(14, 5))
plt.plot(y_test)
plt.plot(y_pred)
plt.show()
