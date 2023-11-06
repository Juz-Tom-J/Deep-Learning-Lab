import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
X_train = pad_sequences(X_train, maxlen=200)
X_test = pad_sequences(X_test, maxlen=200)

model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=200),
    LSTM(128),
    Dense(1, activation='sigmoid')
    ])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=64)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")

# Reshape the sequence to match the batch size expected by the model
test_seq = np.reshape(X_test[7], (1, -1))

pred = model.predict(test_seq)[0]
print('Positive Review') if int(pred[0]) == 1 else print('Negative Review')

# Checking the correctness of prediction
y_test[7] # 0 means Negative Review ; 1 means Positive Review
