'''

Title: Affinity Predictor
Author: Anirudh Venkatraman
Availibility: https://github.com/anirudhvenkatraman/synopsys-2021

Class to predict IC-50 score by loading the weights of the CNN and running model.predict() on a
given molecule and a FASTA sequence.

To train the model, run affinity_CNN_train in a jupyter notebook.

'''


import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import backend as K
import numpy as np


class AffinityPrediction:

    def __init__(self):
        self.batch_siz = 105000
        self.max_smiles = 137
        self.max_fasta = 5000

        list_smiles_elements = ['6', '3', '=', 'H', 'C', 'O', 'c', '#', 'a', '[', 't', 'r', 'K', 'n', 'B', 'F', '4', '+', ']', '-', '1',
                                'P', '0', 'L', '%', 'g', '9', 'Z', '(', 'N', '8', 'I', '7', '5', 'l', ')', 'A', 'e', 'o', 'V', 's', 'S', '2', 'M', 'T', 'u', 'i']
        self.smiles_to_int = dict(
            zip(list_smiles_elements, range(1, len(list_smiles_elements)+1)))
        # added one for empty characters filled in with 0's
        self.elements = len(list_smiles_elements) + 1

        amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                       'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
        self.amino_to_int = dict(
            zip(amino_acids, range(1, len(amino_acids)+1)))
        # added one for empty characters filled in with 0's
        self.aminos = len(amino_acids) + 1
        self.model = self.build_and_compile_model()
#         print(self.model.summary())

    def build_and_compile_model(self):
        # 1D CNN model
        smiles_input = layers.Input(
            shape=(self.max_smiles,), dtype='int32', name='smiles_input')

        # encode the input sequence into a sequence of dense n-dimensional vectors
        embed_smiles = layers.Embedding(
            output_dim=128, input_dim=self.elements, input_length=self.max_smiles)(smiles_input)

        # use 1-D convolutional filters to transform the vector sequence into a single vector containing information about the entire sequence
        conv1_smiles = layers.Conv1D(filters=32, kernel_size=3, padding='SAME', input_shape=(
            self.batch_siz, self.max_smiles))(embed_smiles)
        activation1_smiles = layers.PReLU()(conv1_smiles)
        conv2_smiles = layers.Conv1D(
            filters=64, kernel_size=3, padding='SAME')(activation1_smiles)
        activation2_smiles = layers.PReLU()(conv2_smiles)
        conv3_smiles = layers.Conv1D(
            filters=128, kernel_size=3, padding='SAME')(activation2_smiles)
        activation3_smiles = layers.PReLU()(conv3_smiles)
        # create vector for dense layers by applying pooling on the spatial dimensions until each spatial dimension is one
        global_pool_smiles = layers.GlobalMaxPooling1D()(activation3_smiles)

        fasta_input = layers.Input(
            shape=(self.max_fasta,), dtype='int32', name='fasta_input')

        # encode the input sequence into a sequence of dense n-dimensional vectors
        embed_fasta = layers.Embedding(
            output_dim=256, input_dim=self.aminos, input_length=self.max_fasta)(fasta_input)

        # use 1-D convolutional filters to transform the vector sequence into a single vector containing information about the entire sequence
        conv1_fasta = layers.Conv1D(filters=32, kernel_size=3, padding='SAME', input_shape=(
            self.batch_siz, self.max_fasta))(embed_fasta)
        activation1_fasta = layers.PReLU()(conv1_fasta)
        conv2_fasta = layers.Conv1D(
            filters=64, kernel_size=3, padding='SAME')(activation1_fasta)
        activation2_fasta = layers.PReLU()(conv2_fasta)
        conv3_fasta = layers.Conv1D(
            filters=128, kernel_size=3, padding='SAME')(activation2_fasta)
        activation3_fasta = layers.PReLU()(conv3_fasta)
        # create vector for dense layers by applying pooling on the spatial dimensions until each spatial dimension is one
        global_pool_fasta = layers.GlobalMaxPooling1D()(activation3_fasta)

        # merge both smiles and fasta
        concat_pools = layers.concatenate(
            [global_pool_smiles, global_pool_fasta])

        # dense layers
        dense1 = layers.Dense(1024, activation='relu')(concat_pools)
        dropout1_dense = layers.Dropout(0.1)(dense1)
        dense2 = layers.Dense(512, activation='relu')(dropout1_dense)

        # output
        # relu range (0, inf) --> matches labels because the data only contains positive IC50 values
        output = layers.Dense(1, name='output', activation="relu",
                              kernel_initializer="normal")(dense2)

        model = Model(inputs=[smiles_input, fasta_input], outputs=[output])

        # loss function

        def root_mean_squared_error(y_true, y_pred):
            return K.sqrt(mean_squared_error(y_true, y_pred))

        # accuracy metric

        def r2_score(y_true, y_pred):
            SS_res = K.sum(K.square(y_true - y_pred))
            SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
            return (1-SS_res/(SS_tot)+K.epsilon())

        # compile model
        model.compile(optimizer='adam',
                      # 'mse', 'mae', root_mean_squared_error
                      loss={'output': 'mae'},
                      metrics={'output': r2_score})

        model.load_weights("affinityCNN/weights/affinity-best (9).hdf5")
        return(model)

    def predict_affinity(self, smiles, fasta):

        smiles_in = []
        for element in smiles:
            smiles_in.append(self.smiles_to_int[element])
        while(len(smiles_in) != self.max_smiles):
            smiles_in.append(0)

        fasta_in = []
        for amino in fasta:
            fasta_in.append(self.amino_to_int[amino])
        while(len(fasta_in) != self.max_fasta):
            fasta_in.append(0)

        return self.model({'smiles_input': np.array(smiles_in).reshape(1, self.max_smiles,),
                           'fasta_input': np.array(fasta_in).reshape(1, self.max_fasta,)}, training=False)[0][0]
