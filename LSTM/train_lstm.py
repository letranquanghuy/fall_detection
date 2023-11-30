import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# Đọc dữ liệu
fall_dataset = pd.read_csv("FALL.txt")
not_fall_dataset = pd.read_csv("NOT_FALL.txt")

X = []
y = []
no_of_timesteps = 20

not_fall_data = not_fall_dataset.iloc[:,1:].values
n_sample = len(not_fall_data)
for i in range(no_of_timesteps, n_sample):
    X.append(not_fall_data[i-no_of_timesteps:i,:])
    y.append(1)

fall_data = fall_dataset.iloc[:,1:].values
n_sample = len(fall_data)
for i in range(no_of_timesteps, n_sample):
    X.append(fall_data[i-no_of_timesteps:i,:])
    y.append(1)


X, y = np.array(X), np.array(y)
X, y = shuffle(X, y)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model  = Sequential()
model.add(LSTM(units = 256, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 128, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 128, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1, activation="sigmoid"))
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "binary_crossentropy")

model.fit(X_train, y_train, epochs=16, batch_size=64, validation_data=(X_test, y_test))
model.save("model.h5")
