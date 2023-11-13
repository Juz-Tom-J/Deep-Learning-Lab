import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

nifty = "nifty.csv"
data = pd.read_csv(nifty, index_col="Date", parse_dates=True)

# Normalize numerical columns
scaler = MinMaxScaler()
data[['Open', 'High', 'Low', 'Close']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close']])

# Ensure data is not shuffled while splitting
train_data, test_data = train_test_split(data, test_size=0.3, shuffle=False)

model = tf.keras.Sequential([
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(loss='mse', optimizer='adam')
model.fit(train_data[['Open', 'High', 'Low']], train_data['Close'], epochs=40)
predicted_closing_prices = model.predict(test_data[['Open', 'High', 'Low']])

plt.plot(test_data.index, test_data['Close'], label='Actual Closing Price')
plt.plot(test_data.index, predicted_closing_prices, label='Predicted Closing Price')
plt.title("Closing Price Distribution")
plt.xlabel("Date")
plt.legend()
plt.show()

mae = mean_absolute_error(test_data['Close'], predicted_closing_prices)
print(f"\nMean Absolute Error : {round(mae, 5)}")
