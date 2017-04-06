"""Sequence-to-Sequence word2vec."""
import numpy as np

import keras.backend as K
import keras.models
from keras.optimizers import RMSprop
from keras.layers import Input, Conv3D
from keras.layers import LSTM, RepeatVector, Input, Reshape
from keras.layers.core import Masking, Dense, Flatten, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import MaxPooling3D
from keras.layers import merge
from keras.models import Model
from keras import regularizers
from sklearn.preprocessing import normalize

from .base import BaseSeq2Vec
from .base import TrainableInterfaceMixin
from .base import BaseTransformer
from .seq2seq_word2vec import Seq2vecWord2vecSeqTransformer

def _create_cnn3D_auto_encoder_model(
        max_length,
        embedding_size,
        conv_size,
        latent_size,
        learning_rate,
        channel_size,
        rho=0.9,
        decay=0.01,
    ):

    input_img = Input(shape=(max_length, max_length, embedding_size, 1))

    final_window_size = max_length - 1
    final_feature_size = embedding_size // conv_size
    final_feature_window_size = 1

    x = Conv3D(
        channel_size, (2, 2, conv_size), strides=(1, 1, conv_size),
        activation='tanh', padding='valid', use_bias=False,
        kernel_regularizer=regularizers.l2(0.001)
    )(input_img)

    mask_input = Input(
        shape=(
            final_window_size, final_window_size,
            final_feature_size, channel_size
        )
    )
    mask_x = merge([x, mask_input], mode='sum')

    mask_x = MaxPooling3D(
        (final_window_size, final_window_size, final_feature_window_size),
        padding='valid'
    )(mask_x)

    encoded_output = Flatten()(mask_x)
    decoder_input = RepeatVector(max_length)(encoded_output)

    de_LSTM = LSTM(
        latent_size, return_sequences=True, implementation=2,
        name='de_LSTM_1', unroll=False, dropout=0.1, recurrent_dropout=0.1,
        kernel_regularizer=regularizers.l2(0.001),
        recurrent_regularizer=regularizers.l2(0.001)
    )(decoder_input)

    dense_input = Dropout(0.1)(de_LSTM)
    dense_output = TimeDistributed(
        Dense(
            embedding_size, name='output',
            kernel_regularizer=regularizers.l2(0.001),
            activation='tanh'
        )
    )(dense_input)

    mask_output = Input(shape=(max_length, embedding_size))

    output = merge([dense_output, mask_output], mode='mul')

    model = Model([input_img, mask_input, mask_output], output)
    encoder = Model([input_img, mask_input, mask_output], encoded_output)

    optimizer = RMSprop(
        lr=learning_rate,
        rho=rho,
        decay=decay,
    )
    model.compile(loss='cosine_proximity', optimizer=optimizer)
    return model, encoder

class Seq2vecCNN3DTransformer(BaseTransformer):

    def __init__(self, word2vec_model, max_length, conv_size, channel_size):
        self.max_length = max_length
        self.embedding_size = word2vec_model.get_size()
        self.word2vec = word2vec_model
        self.mask_feature_size = self.embedding_size // conv_size
        self.mask_window_size = self.max_length - 1
        self.channel_size = channel_size

    def seq_transform(self, seq):
        transformed_seq = []
        for word in seq:
            try:
                word_arr = self.word2vec[word]
                normalize(word_arr.reshape(1, -1), copy=False)
                transformed_seq.append(word_arr.reshape(self.embedding_size))
            except KeyError:
                pass

        transformed_array = np.zeros((
            self.max_length, self.max_length, self.embedding_size
        ))

        seq_length = len(transformed_seq)
        if seq_length > self.max_length:
            seq_length = self.max_length

        for i in range(seq_length):
            for j in range(seq_length):
                if i > j:
                    transformed_array[i, j, :] = transformed_array[j, i, :]
                else:
                    transformed_array[i, j, :] = (
                        transformed_seq[i] + transformed_seq[j]
                    ) / 2

        return seq_length, transformed_array.reshape(
            self.max_length, self.max_length, self.embedding_size, 1
        )

    def gen_input_mask(self, seq_length):
        if seq_length > self.max_length:
            seq_length = self.max_length

        mask_input = np.zeros(
            shape=(
                self.mask_window_size, self.mask_window_size,
                self.mask_feature_size, self.channel_size
            )
        )
        if seq_length < self.max_length:
            mask_input[seq_length:, :, :, :] = -10.0
            mask_input[:, seq_length:, :, :] = -10.0
        return mask_input

    def gen_output_mask(self, seq_length):
        if seq_length > self.max_length:
            seq_length = self.max_length

        mask_output = np.ones(
            shape=(
                self.max_length, self.embedding_size
            )
        )
        if seq_length < self.max_length:
            mask_output[seq_length:, :] = 0.0
        return mask_output

    def __call__(self, seqs):
        array_list = []
        input_mask_list = []
        output_mask_list = []
        for seq in seqs:
            seq_length, transformed_array = self.seq_transform(seq)
            array_list.append(transformed_array)
            input_mask_list.append(self.gen_input_mask(seq_length))
            output_mask_list.append(self.gen_output_mask(seq_length))
        return [
            np.array(array_list), np.array(input_mask_list),
            np.array(output_mask_list)
        ]

class Seq2SeqCNN(TrainableInterfaceMixin, BaseSeq2Vec):
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
            max_length=10,
            latent_size=300,
            learning_rate=0.0001,
            conv_size=5,
            channel_size=10,
        ):
        self.word2vec_model = word2vec_model
        self.input_transformer = Seq2vecCNN3DTransformer(
            word2vec_model, max_length, conv_size, channel_size
        )
        self.output_transformer = Seq2vecWord2vecSeqTransformer(
            word2vec_model, max_length, inverse=False
        )
        self.embedding_size = word2vec_model.get_size()
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.conv_size = conv_size
        self.latent_size = latent_size
        self.channel_size = channel_size
        self.encoding_size = (
            self.embedding_size // self.conv_size * self.channel_size
        )

        model, encoder = _create_cnn3D_auto_encoder_model(
            max_length=self.max_length,
            embedding_size=self.embedding_size,
            conv_size=self.conv_size,
            channel_size=channel_size,
            learning_rate=self.learning_rate,
            latent_size=self.latent_size,
        )
        self.model = model
        self.encoder = encoder

    def load_model(self, file_path):
        self.model = keras.models.load_model(file_path)
        encoded_output = self.model.get_layer(index=5).output
        self.encoder = Model(
            self.model.input, encoded_output
        )
        self.embedding_size = self.model.input_shape[0][3]
        self.max_length = self.model.input_shape[0][1]
        self.conv_size = self.embedding_size // self.model.input_shape[1][3]
        self.latent_size = self.model.get_layer(index=9).input_shape[2]
        self.channel_size = self.model.input_shape[1][4]
        self.encoding_size = self.encoder.output_shape[1]

        self.input_transformer = Seq2vecCNN3DTransformer(
            self.word2vec_model, self.max_length,
            self.conv_size, self.channel_size
        )

        self.output_transformer = Seq2vecWord2vecSeqTransformer(
            self.word2vec_model, self.max_length, inverse=False
        )
