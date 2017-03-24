"""Sequence-to-Sequence char2vec."""
import numpy as np

import keras.models
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import LSTM, RepeatVector
from keras.layers.core import Masking
from keras.models import Model
from sklearn.preprocessing import normalize

from .base import BaseSeq2Vec
from .base import TrainableInterfaceMixin
from .base import BaseTransformer
from .util import generate_padding_array

def _create_single_layer_seq2seq_model(
        max_length,
        max_index,
        latent_size,
        learning_rate,
        rho=0.9,
        decay=0.01,
    ):
    model = Sequential()
    model.add(
        Masking(mask_value=0.0, input_shape=(max_length, max_index))
    )
    model.add(
        LSTM(
            latent_size, return_sequences=False, name='en_LSTM_1',
            dropout=0.2, recurrent_dropout=0.3
        )
    )
    model.add(RepeatVector(max_length))
    model.add(
        LSTM(
            max_index, return_sequences=True, name='de_LSTM_1',
            dropout=0.2, recurrent_dropout=0.3
        )
    )
    encoder = Model(model.input, model.get_layer('en_LSTM_1').output)

    optimizer = RMSprop(
        lr=learning_rate,
        rho=rho,
        decay=decay,
    )
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model, encoder

class Seq2vecChar2vecSeqTransformer(BaseTransformer):

    def __init__(self, char2vec_model, max_length, inverse):
        self.max_length = max_length
        self.max_index = char2vec_model.get_size()
        self.char2vec = char2vec_model
        self.inverse = inverse

    def seq_transform(self, seq):
        seq = ''.join(seq)
        transformed_seq = []
        for word in seq:
            try:
                word_arr = self.char2vec[word]
                normalize(word_arr.reshape(1, -1), copy=False)
                transformed_seq.append(word_arr.reshape(self.max_index))
            except KeyError:
                pass
        return transformed_seq

    def __call__(self, seqs):
        array = generate_padding_array(
            seqs, self.seq_transform, np.zeros(self.max_index),
            self.max_length, inverse=self.inverse
        )
        return array

class Seq2SeqChar2Vec(TrainableInterfaceMixin, BaseSeq2Vec):
    """seq2seq auto-encoder using pretrained word vectors as input.

    Attributes
    ----------
    max_index: int
        The length of input feature

    max_length: int
        The length of longest sequence.

    latent_size: int
        The returned latent vector size after encoding.

    """

    def __init__(
            self,
            char2vec_model,
            max_length,
            learning_rate=0.0001,
            latent_size=20,
        ):
        self.input_transformer = Seq2vecChar2vecSeqTransformer(
            char2vec_model, max_length, True
        )
        self.output_transformer = Seq2vecChar2vecSeqTransformer(
            char2vec_model, max_length, False
        )
        self.max_index = char2vec_model.get_size()
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.latent_size = latent_size

        model, encoder = _create_single_layer_seq2seq_model(
            max_length=self.max_length,
            max_index=self.max_index,
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
        self.max_index = self.model.input_shape[2]
        self.max_length = self.model.input_shape[1]
        self.latent_size = self.model.get_layer('en_LSTM_1').output_dim
