"""Sequence-to-Sequence word2vec."""
import numpy as np

import keras.models
from keras.optimizers import RMSprop
from keras.layers import Input, Reshape
from keras.layers.core import Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.models import Model
from keras import regularizers
from sklearn.preprocessing import normalize

from yoctol_utils.hash import consistent_hash
from yklz import MaskConv, ConvEncoder, MaskConvNet
from yklz import MaskToSeq, MaskPooling
from yklz import RNNDecoder, LSTMPeephole, RNNCell
from .base import BaseSeq2Vec
from .base import TrainableInterfaceMixin
from .base import BaseTransformer

def _create_char2vec_auto_encoder_model(
        max_index,
        max_length,
        embedding_size,
        word_embedding_size,
        conv_size,
        learning_rate,
        latent_size,
        channel_size,
        rho=0.9,
        decay=0.01,
    ):

    inputs = Input(shape=(max_length, max_index))
    char_embedding = TimeDistributed(
        Dense(
            embedding_size, use_bias=False,
            kernel_regularizer=regularizers.l2(0.001),
            activation='tanh'
        )
    )(inputs)

    char_embedding = Reshape((max_length, embedding_size, 1))(char_embedding)
    masked_embedding = MaskConv(0.0)(char_embedding)
    masked_seq = MaskToSeq(
        layer=MaskConv(0.0),
        time_axis=1
    )(char_embedding)

    char_feature = MaskConvNet(
        Conv2D(
            channel_size, (2, conv_size), strides=(1, conv_size),
            activation='tanh', padding='valid', use_bias=False,
            kernel_regularizer=regularizers.l2(0.001)
        )
    )(masked_embedding)

    final_window_size = max_length - 1
    final_feature_size = channel_size * embedding_size // conv_size

    mask_feature = MaskPooling(
        MaxPool2D(
            (final_window_size, 1),
            padding='valid'
        ),
        pool_mode='max'
    )(char_feature)

    encoded_feature = ConvEncoder()([mask_feature, masked_seq])

    dense_input = RNNDecoder(
        RNNCell(
            LSTMPeephole(
                units=latent_size,
                return_sequences=True,
                implementation=2,
                unroll=False,
                dropout=0.1,
                recurrent_dropout=0.1,
                kernel_regularizer=regularizers.l2(0.001),
                recurrent_regularizer=regularizers.l2(0.001),
            ),
            Dense(
                units=final_feature_size,
                activation='tanh'
            ),
            dense_dropout=0.1
        )
    )(encoded_feature)

    outputs = TimeDistributed(
        Dense(
            word_embedding_size,
            kernel_regularizer=regularizers.l2(0.001),
            activation='tanh'
        )
    )(dense_input)

    model = Model(inputs, outputs)
    encoder = Model(inputs, encoded_feature)

    optimizer = RMSprop(
        lr=learning_rate,
        rho=rho,
        decay=decay,
    )
    model.compile(loss='cosine_proximity', optimizer=optimizer)
    return model, encoder

class Seq2vecChar2vecInputTransformer(BaseTransformer):

    def __init__(
        self,
        word2vec,
        max_index,
        max_length,
        embedding_size,
        conv_size,
        channel_size
    ):
        self.word2vec = word2vec
        self.word_embedding_size = self.word2vec.get_size()
        self.max_index = max_index
        self.max_length = max_length
        self.embedding_size = embedding_size
        self.mask_feature_size = self.embedding_size // conv_size
        self.mask_window_size = self.max_length - 1
        self.channel_size = channel_size

    def seq_transform(self, seq):
        seq = ''.join(seq)

        seq_length = len(seq)
        if seq_length > self.max_length:
            seq_length = self.max_length

        transformed_array = np.zeros((
            self.max_length, self.max_index
        ))

        for i in range(seq_length):
            char = seq[i]
            index = consistent_hash(char) % self.max_index
            transformed_array[i, index] = 1.0

        return seq_length, transformed_array

    def __call__(self, seqs):
        array_list = []
        for seq in seqs:
            _, transformed_array = self.seq_transform(seq)
            array_list.append(transformed_array)
        return np.array(array_list)

class Seq2vecChar2vecOutputTransformer(BaseTransformer):

    def __init__(
        self,
        word2vec,
        max_index,
        max_length,
        embedding_size,
        conv_size,
        channel_size
    ):
        self.word2vec = word2vec
        self.word_embedding_size = self.word2vec.get_size()
        self.max_index = max_index
        self.max_length = max_length
        self.embedding_size = embedding_size
        self.mask_feature_size = self.embedding_size // conv_size
        self.mask_window_size = self.max_length - 1
        self.channel_size = channel_size

    def seq_transform(self, seq):
        transformed_array = np.zeros((
            self.max_length, self.word_embedding_size
        ))

        seq_length = 0
        for _, word in enumerate(seq):
            if seq_length < self.max_length:
                try:
                    word_arr = self.word2vec[word]
                    normalize(word_arr.reshape(1, -1), copy=False)
                    transformed_array[seq_length, :] = word_arr.reshape(
                        self.word_embedding_size
                    )
                    seq_length = seq_length + 1
                except KeyError:
                    pass
        return transformed_array

    def __call__(self, seqs):
        array_list = []
        for seq in seqs:
            transformed_array = self.seq_transform(seq)
            array_list.append(transformed_array)
        return np.array(array_list)

class Seq2SeqChar2vec(TrainableInterfaceMixin, BaseSeq2Vec):
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
            word2vec_model,
            max_index=10000,
            max_length=10,
            embedding_size=300,
            learning_rate=0.0001,
            conv_size=5,
            channel_size=10,
            latent_size=20,
        ):
        self.word2vec = word2vec_model
        self.input_transformer = Seq2vecChar2vecInputTransformer(
            word2vec_model, max_index, max_length, embedding_size, conv_size, channel_size
        )
        self.output_transformer = Seq2vecChar2vecOutputTransformer(
            word2vec_model, max_index, max_length, embedding_size, conv_size, channel_size
        )
        self.embedding_size = embedding_size
        self.max_length = max_length
        self.max_index = max_index
        self.learning_rate = learning_rate
        self.conv_size = conv_size
        self.channel_size = channel_size
        self.latent_size = latent_size
        self.encoding_size = (
            self.embedding_size // self.conv_size * self.channel_size
        )

        model, encoder = _create_char2vec_auto_encoder_model(
            max_index=self.max_index,
            max_length=self.max_length,
            embedding_size=self.embedding_size,
            word_embedding_size=self.word2vec.get_size(),
            conv_size=self.conv_size,
            channel_size=channel_size,
            learning_rate=self.learning_rate,
            latent_size=self.latent_size
        )
        self.model = model
        self.encoder = encoder

    def transform(self, seqs):
        test_x = self.input_transformer(seqs)
        return self.encoder.predict(test_x)[:, 0, :]

    def load_customed_model(self, file_path):
        return keras.models.load_model(
            file_path, custom_objects={
                'RNNDecoder': RNNDecoder,
                'MaskPooling': MaskPooling,
                'MaskToSeq': MaskToSeq,
                'MaskConv': MaskConv,
                'MaskConvNet': MaskConvNet,
                'ConvEncoder': ConvEncoder,
                'LSTMPeephole':LSTMPeephole,
                'RNNCell':RNNCell,
            }
        )

    def load_model(self, file_path):
        self.model = self.load_customed_model(file_path)
        encoded_output = self.model.get_layer(index=7).output
        self.encoder = Model(
            self.model.input, encoded_output
        )
        self.embedding_size = self.model.get_layer(index=1).output_shape[2]
        self.max_length = self.model.get_layer(index=0).output_shape[1]
        self.max_index = self.model.input_shape[2]
        self.conv_size = self.embedding_size // self.model.get_layer(index=4).output_shape[2]
        self.channel_size = self.model.get_layer(index=4).output_shape[3]
        self.encoding_size = self.encoder.output_shape[2]
        self.latent_size = self.model.get_layer(index=8).layer.recurrent_layer.units

        self.input_transformer = Seq2vecChar2vecInputTransformer(
            self.word2vec, self.max_index, self.max_length, self.embedding_size,
            self.conv_size, self.channel_size
        )
        self.output_transformer = Seq2vecChar2vecOutputTransformer(
            self.word2vec, self.max_index, self.max_length, self.embedding_size,
            self.conv_size, self.channel_size
        )
