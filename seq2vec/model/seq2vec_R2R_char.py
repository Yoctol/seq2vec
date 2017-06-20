"""Sequence-to-Sequence Auto Encoder."""
import numpy as np

import keras.models
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import LSTM, RepeatVector
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense, Dropout, Activation
from keras.models import Model

from seq2vec.transformer import HashIndexTransformer
from seq2vec.transformer import OneHotEncodedTransformer
from seq2vec.model import Seq2VecBase
from seq2vec.model import TrainableInterfaceMixin

def _create_single_layer_seq2seq_model(
        max_length,
        max_index,
        embedding_size,
        latent_size,
        learning_rate,
        rho=0.9,
        decay=0.01,
    ):

    model = Sequential()
    model.add(
        Embedding(
            max_index, embedding_size, input_length=max_length,
            name='embedding', mask_zero=True, dropout=0.2
        )
    )
    model.add(
        LSTM(
            units=latent_size, return_sequences=False,
            name='en_LSTM_1', dropout=0.2, recurrent_dropout=0.3
        )
    )
    model.add(
        RepeatVector(max_length)
    )
    model.add(
        LSTM(
            embedding_size, return_sequences=True,
            name='de_LSTM_1', dropout=0.2, recurrent_dropout=0.3
        )
    )
    model.add(
        TimeDistributed(Dense(max_index))
    )
    model.add(Dropout(0.2))
    model.add(Activation('softmax'))

    encoder = Model(
        model.input, model.get_layer('en_LSTM_1').output
    )

    optimizer = RMSprop(
        lr=learning_rate,
        rho=rho,
        decay=decay,
    )
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model, encoder


class Seq2SeqAutoEncoderUseWordHash(TrainableInterfaceMixin, Seq2VecBase):
    """Hash words and feed to seq2seq auto-encoder.

    Attributes
    ----------
    max_index: int
        The length of input vector.

    max_length: int
        The length of longest sequence.

    latent_size: int
        The returned latent vector size after encoding.

    """

    def __init__(
            self,
            max_index,
            max_length,
            learning_rate=0.0001,
            embedding_size=64,
            latent_size=20,
        ):
        self.max_index = max_index
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.latent_size = latent_size

        self.input_transformer = HashIndexTransformer(
            max_index, max_length
        )
        self.output_transformer = OneHotEncodedTransformer(
            max_index, max_length
        )

        model, encoder = _create_single_layer_seq2seq_model(
            max_length=self.max_length,
            max_index=self.max_index + 1,
            embedding_size=self.embedding_size,
            latent_size=self.latent_size,
            learning_rate=self.learning_rate,
        )
        self.model = model
        self.encoder = encoder

    def load_model(self, file_path):
        self.model = keras.models.load_model(file_path)
        self.encoder = Model(
            self.model.input, self.model.get_layer('en_LSTM_1').output
        )
        self.max_index = self.model.get_layer('embedding').input_dim - 1
        self.max_length = self.model.input_shape[1]
        self.embedding_size = self.model.get_layer('embedding').output_dim
        self.latent_size = self.model.get_layer('en_LSTM_1').units
