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
import matplotlib as plt
from pandas.core.common import flatten
import numpy as np
import time
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import Draw


class GenerateMolecules:

    def split_input_target(self, chunk):
        input_text = chunk[:-1]
        target_idx = chunk[-1]
        target = tf.one_hot(target_idx, depth=34)
        target = tf.reshape(target, [-1])
        return input_text, target

    def preprocess_data(self):
        path_to_file = "generation-RNN/data/100k_SMILES.txt"
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

        dataset = sequences.map(self.split_input_target)

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

    def build_and_compile_model(self):
        reverse_map = pickle.load(
            open("generation-RNN/saved-variables/char_map.pkl", mode="rb"))
        char_map = pickle.load(
            open("generation-RNN/saved-variables/reverse_map.pkl", mode="rb"))

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

        model.load_weights(
            "generation-RNN/weights/generation_RNN_weights.hdf5")
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        return(model, char_map, reverse_map)

    def generate(self, iterations):
        t = time.localtime()
        # file = open("generation-RNN/generated-molecules/" +timestamp + ".txt", "a")
        model, char_map, reverse_map = self.build_and_compile_model()
        generated_mols = ''

        for i in range(iterations):
            cur_mol = ''

            network_input = tfds.as_numpy(
                self.preprocess_data().take(np.random.randint(0, 30000)))
            for i, x in enumerate(network_input):
                break

            pattern = x[0][np.random.randint(0, 127)]
            print("Seed:")
            print("\"", ''.join([char_map[value[0]]
                                 for value in pattern]), "\"")

            for i in range(137):
                x = np.reshape(pattern, (1, len(pattern), 1))
                prediction = model.predict(x, verbose=0)
                index = np.argmax(prediction)
                result = char_map[index]
                cur_mol += result
                pattern = np.append(pattern, np.array([[index]]), axis=0)
                pattern = pattern[1:len(pattern)]

            generated_mols += cur_mol + "\n"

        final_mols = self.validateMols(generated_mols)
        return (final_mols)

    def isValid(self, mol):
        if mol == None or len(mol) <= 3:
            return False
        mol = Chem.MolFromSmiles(mol)
        if mol == None:
            return False
        return (True)

    def get_h_bond_donors(self, mol):
        idx = 0
        donors = 0
        while idx < len(mol)-1:
            if mol[idx].lower() == "o" or mol[idx].lower() == "n":
                if mol[idx+1].lower() == "h":
                    donors += 1
            idx += 1
        return donors

    def get_h_bond_acceptors(self, mol):
        acceptors = 0
        for i in mol:
            if i.lower() == "n" or i.lower() == "o":
                acceptors += 1
        return acceptors

    def isDrugLike(self, mol):
        m = Chem.MolFromSmiles(mol)
        if self.get_h_bond_donors(mol) <= 5 and self.get_h_bond_acceptors(mol) <= 10 and Descriptors.MolWt(m) <= 500 and Descriptors.MolLogP(m) <= 5:
            return True
        else:
            return False

    def validateMols(self, data):
        data = [data[y - 138:y] for y in range(138, len(data) + 138, 138)]
        mol = [seq.split("\n") for seq in data]
        mol = [list(filter(None, arr)) for arr in mol]
        mol = [i[1:-1] for i in mol]
        mol = list(flatten(mol))

        mol_list = []

        validMol = 0
        invalidMol = 0
        for m in mol:
            if (self.isValid(m)):
                mol_list.append(m)

        return(mol_list)
