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
import pandas

tf.compat.v1.enable_eager_execution()


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_idx = chunk[-1]
    target = tf.one_hot(target_idx, depth=34)
    target = tf.reshape(target, [-1])
    return input_text, target


def preprocess_data():
    path_to_file = "./100k_SMILES.txt"
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


def get_compiled_model():
    model = Sequential(
        [
            LSTM(128, input_shape=(137, 1), return_sequences=True),
            Dropout(0.3),
            LSTM(256, return_sequences=True),
            Dropout(0.3),
            LSTM(512, return_sequences=True),
            Dropout(0.3),
            LSTM(256, return_sequences=True),
            Dropout(0.3),
            LSTM(128),
            Dropout(0.3),
            Dense(34, activation="softmax")
        ]
    )

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    return(model)


filepath = "./weights_2a/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    model = get_compiled_model()


# Train the model on all available devices.
train_dataset = preprocess_data()
history = model.fit(train_dataset, epochs=10)
