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
from keras.models import Model
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
        rho=0.9,
        decay=0.01,
    ):

    input_img = Input(shape=(max_length, max_length, embedding_size, 1))

    x = Conv3D(32, (2, 2, conv_size), activation='relu', padding='valid')(input_img)
    #x = MaxPooling3D((2, 2, 5), padding='valid')(x)
    x = Conv3D(16, (2, 2, conv_size), activation='relu', padding='valid')(x)
    #x = MaxPooling3D((2, 2, 5), padding='valid')(x)
    encoded = Conv3D(8, (2, 2, conv_size), activation=None, padding='valid')(x)

    encoded_output = Flatten()(encoded)
    decoder_input = RepeatVector(max_length)(encoded_output)

    de_LSTM = LSTM(
        latent_size, return_sequences=True, implementation=2,
        name='de_LSTM_1', unroll=False, dropout=0.2, recurrent_dropout=0.3
    )(decoder_input)
    dense_input = Dropout(0.1)(de_LSTM)
    output = TimeDistributed(Dense(embedding_size, name='output'))(dense_input)

    model = Model(input_img, output)
    encoder = Model(input_img, encoded_output)

    optimizer = RMSprop(
        lr=learning_rate,
        rho=rho,
        decay=decay,
    )
    model.compile(loss='cosine_proximity', optimizer=optimizer)
    return model, encoder

class Seq2vecCNN3DTransformer(BaseTransformer):

    def __init__(self, word2vec_model, max_length):
        self.max_length = max_length
        self.embedding_size = word2vec_model.get_size()
        self.word2vec = word2vec_model

    def seq_transform(self, seq):
        transformed_seq = []
        for word in seq:
            try:
                word_arr = self.word2vec[word]
                normalize(word_arr.reshape(1, -1), copy=False)
                transformed_seq.append(word_arr.reshape(self.max_index))
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

        return transformed_array.reshape(
            self.max_length, self.max_length, self.embedding_size, 1
        )

    def __call__(self, seqs):
        array_list = []
        for seq in seqs:
            array_list.append(self.seq_transform(seq))
        return np.array(array_list)

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
            max_length,
            latent_size=300,
            learning_rate=0.0001,
            conv_size=5,
        ):
        self.input_transformer = Seq2vecCNNTransformer(
            word2vec_model, max_length
        )
        self.output_transformer = Seq2vecWord2vecSeqTransformer(
            word2vec_model, max_length, inverse=False
        )
        self.embedding_size = word2vec_model.get_size()
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.conv_size = conv_size
        self.latent_size = latent_size

        model, encoder = _create_cnn3D_auto_encoder_model(
            max_length=self.max_length,
            embedding_size=self.embedding_size,
            conv_size=self.conv_size,
            learning_rate=self.learning_rate,
            latent_size=self.latent_size
        )
        self.model = model
        self.encoder = encoder

    def load_model(self, file_path):
        self.model = keras.models.load_model(file_path)
        encoded_output = self.model.get_layer(index=5).output
        self.encoder = Model(
            self.model.input, encoded_output
        )
        self.embedding_size = self.model.input_shape[3]
        self.max_length = self.model.input_shape[1]
