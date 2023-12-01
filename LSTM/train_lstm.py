import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def custom_optimizer_init_fn():
    learning_rate = 1e-5
    return tf.keras.optimizers.Adam(learning_rate)

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      # keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

# Đọc dữ liệu
fall_dataset = pd.read_csv("D:/HCMUT/Ths/Thesis/LSTM/data/FALL.csv")
not_fall_dataset = pd.read_csv("D:/HCMUT/Ths/Thesis/LSTM/data/NOT_FALL.csv")

data = []
label = []
no_of_timesteps = 10

not_fall_data = not_fall_dataset.iloc[:,1:].values
n_sample = len(not_fall_data)
for i in range(no_of_timesteps, n_sample):
    data.append(not_fall_data[i-no_of_timesteps:i,:])
    label.append([1,0])

fall_data = fall_dataset.iloc[:,1:].values
n_sample = len(fall_data)
for i in range(no_of_timesteps, n_sample):
    data.append(fall_data[i-no_of_timesteps:i,:])
    label.append([0,1])


data, label = np.array(data), np.array(label)
data, label = shuffle(data, label)
print(data.shape, label.shape)
data_train, data_val, label_train, label_val = train_test_split(data, label, test_size=0.25)

model  = Sequential()
model.add(LSTM(units = 256, return_sequences = True, input_shape = (data.shape[1], data.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 128, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 10, activation="sigmoid"))
model.add(Dense(units = 2, activation="softmax"))
optimize = custom_optimizer_init_fn()
# model.compile(optimizer=optimize, metrics = ['accuracy'], loss = "binary_crossentropy")
model.compile(optimizer=optimize, metrics = METRICS, loss = "binary_crossentropy")
model.summary ()
# model.fit(X_train, y_train, epochs=150, batch_size=300, validation_data=(X_test, y_test))
batch_size = 500
epochs = 300
model.fit(data_train, label_train, batch_size=batch_size, epochs=epochs, validation_data=(data_val, label_val))
model.save("D:/HCMUT/Ths/Thesis/LSTM/output/model.h5")
