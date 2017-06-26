"""Sequence-to-Sequence Auto Encoder."""
import keras.models
from keras.models import Input
from keras.optimizers import RMSprop
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense, LSTM
from keras.models import Model

from yklz import BidirectionalRNNEncoder, RNNDecoder
from yklz import RNNCell, Pick

from seq2vec.transformer import HashIndexTransformer
from seq2vec.transformer import OneHotEncodedTransformer
from seq2vec.model import TrainableSeq2VecBase


class Seq2VecR2RHash(TrainableSeq2VecBase):
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
            max_index=300,
            max_length=10,
            encoding_size=100,
            learning_rate=0.0001,
            word_embedding_size=64,
            latent_size=20,
            **kwargs
        ):
        self.max_index = max_index
        self.word_embedding_size = word_embedding_size
        self.encoding_size = encoding_size

        self.input_transformer = HashIndexTransformer(
            max_index, max_length
        )
        self.output_transformer = OneHotEncodedTransformer(
            max_index, max_length
        )

        super(Seq2VecR2RHash, self).__init__(
            max_length,
            latent_size,
            learning_rate
        )
        self.custom_objects['BidirectionalRNNEncoder']= BidirectionalRNNEncoder
        self.custom_objects['RNNDecoder'] = RNNDecoder
        self.custom_objects['RNNCell'] = RNNCell
        self.custom_objects['Pick'] = Pick

    def create_model(
            self,
            rho=0.9,
            decay=0.0,
        ):

        inputs = Input(
            shape=(
                self.max_length,
            )
        )
        char_embedding = Embedding(
            input_dim=self.max_index,
            output_dim=self.word_embedding_size,
            input_length=self.max_length,
            mask_zero=True,
        )(inputs)

        encoded = BidirectionalRNNEncoder(
            RNNCell(
                LSTM(
                    units=self.latent_size,
                    dropout=0.,
                    recurrent_dropout=0.
                ),
                Dense(
                    units=self.encoding_size // 2,
                    activation='tanh'
                ),
                dense_dropout=0.
            )
        )(char_embedding)
        decoded = RNNDecoder(
            RNNCell(
                LSTM(
                    units=self.latent_size,
                    dropout=0.,
                    recurrent_dropout=0.
                ),
                Dense(
                    units=self.encoding_size,
                    activation='tanh'
                ),
                dense_dropout=0.
            )
        )(encoded)
        outputs = TimeDistributed(
            Dense(
                units=self.max_index,
                activation='softmax'
            )
        )(decoded)

        model = Model(inputs, outputs)

        picked = Pick()(encoded)
        encoder = Model(inputs, picked)

        optimizer = RMSprop(
            lr=self.learning_rate,
            rho=rho,
            decay=decay,
        )
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model, encoder

    def load_model(self, file_path):
        self.model = self.load_customed_model(file_path)
        picked = Pick()(self.model.get_layer(index=2).output)
        self.encoder = Model(
            self.model.input,
            picked
        )
        self.max_index = self.model.get_layer(index=1).input_dim
        self.max_length = self.model.input_shape[1]
        self.word_embedding_size = self.model.get_layer(index=1).output_dim
        self.latent_size = self.model.get_layer(index=2).layer.recurrent_layer.units
        self.encoding_size = self.model.get_layer(index=2).layer.dense_layer.units * 2

        self.input_transformer = HashIndexTransformer(
            self.max_index,
            self.max_length
        )
        self.output_transformer = OneHotEncodedTransformer(
            self.max_index,
            self.max_length
        )
