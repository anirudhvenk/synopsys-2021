from keras.objectives import categorical_crossentropy
from tensorflow.keras.activations import softmax
import sys
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from numpy.testing import assert_allclose
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import time
import os
import fnmatch
import shutil
import pandas

reverse_map = pickle.load(open("./saved-variables/char_map.pkl", mode="rb"))
char_map = pickle.load(open("./saved-variables/reverse_map.pkl", mode="rb"))

model = Sequential(
    [
        LSTM(128, input_shape=(137, 1), return_sequences=True),
        Dropout(0.1),
        LSTM(256, return_sequences=True),
        Dropout(0.1),
        LSTM(512, return_sequences=True),
        Dropout(0.1),
        LSTM(256, return_sequences=True),
        Dropout(0.1),
        LSTM(128),
        Dropout(0.1),
        Dense(34, activation="softmax")
    ]
)

model.load_weights("weights/weights-improvement-164-0.1369.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam')


t = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M', t)
file = open("./generated-molecules/" + timestamp + ".txt", "a")


seed = "ClC[C@H]1CCCO[C@H]1c1ccc(Cl)s1\nCOC(=O)C1=C(CCl)NC(=O)N[C@@H]1c1cccc(Cl)c1\nC[C@H]1[C@H](C(=O)[O-])CCN1C(=O)c1ccc2ncsc2c1\nO=C(NC12CC3CC(CC("
pattern1d = [reverse_map[c] for c in seed]
pattern = [[i] for i in pattern1d]
print("Seed:")
print("\"", ''.join([char_map[value[0]] for value in pattern]), "\"")


for i in range(137):
    x = np.reshape(pattern, (1, len(pattern), 1))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = char_map[index]
    file.write(result)
    pattern = np.append(pattern, np.array([[index]]), axis=0)
    pattern = pattern[1:len(pattern)]

file.close()
