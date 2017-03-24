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

from yoctol_utils.hash import consistent_hash

from .base import BaseSeq2Vec
from .base import TrainableInterfaceMixin
from .base import BaseTransformer
from .util import generate_padding_array

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


def _one_hot_encode_seq(seq, max_index):
    np_seq = []
    for idx in seq:
        arr = np.zeros(max_index + 1)
        arr[idx] = 1
        np_seq.append(arr)
    return np_seq

def _hash_seq(sequence, max_index):
    return [consistent_hash(word) % max_index + 1 for word in sequence]

class Seq2vecAutoEncoderInputTransformer(BaseTransformer):

    def __init__(self, max_index, max_length):
        self.max_index = max_index
        self.max_length = max_length

    def seq_transform(self, seq):
        return _hash_seq(seq, self.max_index)

    def __call__(self, seqs):
        array = generate_padding_array(
            seqs, self.seq_transform, 0, self.max_length, inverse=True
        )
        return array

class Seq2vecAutoEncoderOutputTransformer(BaseTransformer):

    def __init__(self, max_index, max_length):
        self.max_index = max_index
        self.max_length = max_length

    def seq_transform(self, seq):
        transformed_seq = _one_hot_encode_seq(
            _hash_seq(seq, self.max_index), self.max_index
        )
        return transformed_seq

    def __call__(self, seqs):
        array = generate_padding_array(
            seqs, self.seq_transform, np.zeros(self.max_index + 1),
            self.max_length, inverse=False
        )
        return array

class Seq2SeqAutoEncoderUseWordHash(TrainableInterfaceMixin, BaseSeq2Vec):
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

        self.input_transformer = Seq2vecAutoEncoderInputTransformer(
            max_index, max_length
        )
        self.output_transformer = Seq2vecAutoEncoderOutputTransformer(
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
