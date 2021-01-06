import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from keras.activations import softmax
import sys
import keras.backend as K
from numpy.testing import assert_allclose
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from keras.layers.advanced_activations import LeakyReLU

X = np.load("x.npy")
Y = np.load("y.npy")

char_map = pickle.load(open("char_map.pkl", mode="rb"))
reverse_map = pickle.load(open("reverse_map.pkl", mode="rb"))

seq_len = 40

model = Sequential(
    [
        LSTM(128, input_shape=(seq_len, 1)),
        Dropout(0.3),
        Dense(Y.shape[1], activation="softmax"),
    ]
)

model.load_weights("./weights/weights-improvement-75-0.7012.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam')

file = open("./generated-molecules/gen_1.txt", "a")

for i in range(100):
    # np.random.seed(0)
    start = np.random.randint(0, len(X)-1)
    pattern = X[start]
    # print(pattern.flatten())
    print("Seed:")
    print("\"", ''.join([char_map[value[0]] for value in pattern]), "\"")

    for i in range(40):
        x = np.reshape(pattern, (1, len(pattern), 1))
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = char_map[index]
        file.write(result)
        pattern = np.append(pattern, np.array([[index]]), axis=0)
        pattern = pattern[1:len(pattern)]

file.close()
