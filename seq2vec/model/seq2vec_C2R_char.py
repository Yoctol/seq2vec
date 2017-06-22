"""Sequence-to-Sequence word2vec."""

import keras.models
from keras.optimizers import RMSprop
from keras.layers import Input, Reshape
from keras.layers.core import Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.models import Model
from keras import regularizers

from yklz import MaskConv, ConvEncoder, MaskConvNet
from yklz import MaskToSeq, MaskPooling, Pick
from yklz import RNNDecoder, LSTMPeephole, RNNCell

from seq2vec.transformer import CharEmbeddingOneHotTransformer
from seq2vec.transformer import WordEmbeddingTransformer
from seq2vec.model import TrainableSeq2VecBase


class Seq2VecC2RChar(TrainableSeq2VecBase):
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
            **kwargs
        ):
        self.word2vec = word2vec_model
        self.word_embedding_size = word2vec_model.get_size()
        self.input_transformer = CharEmbeddingOneHotTransformer(
            max_index,
            max_length,
        )
        self.output_transformer = WordEmbeddingTransformer(
            word2vec_model,
            max_length,
        )
        self.embedding_size = embedding_size
        self.max_index = max_index
        self.conv_size = conv_size
        self.channel_size = channel_size
        self.encoding_size = (
            self.embedding_size // self.conv_size * self.channel_size
        )

        super(Seq2VecC2RChar, self).__init__(
            max_length,
            latent_size,
            learning_rate
        )

    def create_model(
            self,
            rho=0.9,
            decay=0.01,
        ):

        inputs = Input(
            shape=(
                self.max_length,
                self.max_index
            )
        )
        char_embedding = TimeDistributed(
            Dense(
                self.embedding_size,
                use_bias=False,
                kernel_regularizer=regularizers.l2(0.001),
                activation='tanh'
            )
        )(inputs)

        char_embedding = Reshape((
            self.max_length,
            self.embedding_size,
            1
        ))(char_embedding)
        masked_embedding = MaskConv(0.0)(char_embedding)
        masked_seq = MaskToSeq(
            layer=MaskConv(0.0),
            time_axis=1
        )(char_embedding)

        char_feature = MaskConvNet(
            Conv2D(
                self.channel_size,
                (2, self.conv_size),
                strides=(1, self.conv_size),
                activation='tanh',
                padding='valid',
                use_bias=False,
                kernel_regularizer=regularizers.l2(0.001)
            )
        )(masked_embedding)

        final_window_size = self.max_length - 1
        final_feature_size = self.channel_size * self.embedding_size // self.conv_size

        mask_feature = MaskPooling(
            MaxPool2D(
                (final_window_size, 1),
                padding='valid'
            ),
            pool_mode='max'
        )(char_feature)

        encoded_feature = ConvEncoder()(
            [mask_feature, masked_seq]
        )

        dense_input = RNNDecoder(
            RNNCell(
                LSTMPeephole(
                    units=self.latent_size,
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
                self.word_embedding_size,
                kernel_regularizer=regularizers.l2(0.001),
                activation='tanh'
            )
        )(dense_input)

        model = Model(inputs, outputs)
        picked = Pick()(encoded_feature)
        encoder = Model(inputs, picked)

        optimizer = RMSprop(
            lr=self.learning_rate,
            rho=rho,
            decay=decay,
        )
        model.compile(loss='cosine_proximity', optimizer=optimizer)
        return model, encoder

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
                'Pick':Pick
            }
        )

    def load_model(self, file_path):
        self.model = self.load_customed_model(file_path)
        picked = Pick()(self.model.get_layer(index=7).output)
        self.encoder = Model(
            self.model.input,
            picked
        )
        self.embedding_size = self.model.get_layer(index=1).output_shape[2]
        self.max_length = self.model.get_layer(index=0).output_shape[1]
        self.max_index = self.model.input_shape[2]
        self.conv_size = self.embedding_size // self.model.get_layer(index=4).output_shape[2]
        self.channel_size = self.model.get_layer(index=4).output_shape[3]
        self.encoding_size = self.encoder.output_shape[1]
        self.latent_size = self.model.get_layer(index=8).layer.recurrent_layer.units

        self.input_transformer = CharEmbeddingOneHotTransformer(
            self.max_index,
            self.max_length,
        )
        self.output_transformer = WordEmbeddingTransformer(
            self.word2vec,
            self.max_length,
        )
