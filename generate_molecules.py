from keras.objectives import categorical_crossentropy
from tensorflow.keras.activations import softmax
import tensorflow_datasets as tfds
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


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_idx = chunk[-1]
    target = tf.one_hot(target_idx, depth=34)
    target = tf.reshape(target, [-1])
    return input_text, target


def preprocess_data():
    path_to_file = "./data/100k_SMILES.txt"
    text = open(path_to_file).read()

    vocab = sorted(set(text))

    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([[char2idx[c]] for c in text])

    # The maximum length sentence you want for a single input in characters
    seq_length = 137
    examples_per_epoch = len(text)//(seq_length+1)

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    dataset = sequences.map(split_input_target)

    # Batch size
    BATCH_SIZE = 128

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE, drop_remainder=True)

    return(dataset)


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


for i in range(100):
    network_input = tfds.as_numpy(
        preprocess_data().take(np.random.randint(0, 30000)))
    for i, x in enumerate(network_input):
        break

    pattern = x[0][np.random.randint(0, 127)]
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

    file.write("\n")

file.close()
