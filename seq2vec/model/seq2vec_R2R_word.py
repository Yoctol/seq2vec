"""Sequence-to-Sequence word embedding and RNN to RNN architecture"""
import keras.models
from keras.optimizers import RMSprop
from keras.layers.core import Masking, Dense
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, Input
from yklz import RNNDecoder, RNNCell
from yklz import BidirectionalRNNEncoder, Pick

from seq2vec.transformer import WordEmbeddingTransformer
from seq2vec.model import TrainableSeq2VecBase


class Seq2VecR2RWord(TrainableSeq2VecBase):
    """seq2seq auto-encoder using pretrained word vectors as input.

    Attributes
    ----------
    word_embedding_size: int
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
            latent_size=20,
            encoding_size=100,
            learning_rate=0.0001,
            **kwargs
        ):
        self.word2vec_model = word2vec_model
        self.input_transformer = WordEmbeddingTransformer(
            word2vec_model,
            max_length
        )
        self.output_transformer = WordEmbeddingTransformer(
            word2vec_model,
            max_length
        )
        self.word_embedding_size = word2vec_model.get_size()
        self.encoding_size = encoding_size

        super(Seq2VecR2RWord, self).__init__(
            max_length=max_length,
            latent_size=latent_size,
            learning_rate=learning_rate
        )
        self.custom_objects['BidirectionalRNNEncoder'] = BidirectionalRNNEncoder
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
                self.word_embedding_size
            )
        )
        masked_inputs = Masking(mask_value=0.0)(inputs)
        encoded_seq = BidirectionalRNNEncoder(
            RNNCell(
                LSTM(
                    units=self.latent_size,
                    use_bias=True,
                    implementation=2,
                    dropout=0.,
                    recurrent_dropout=0.,
                ),
                Dense(
                    units=(self.encoding_size // 2),
                    activation='tanh'
                ),
                dense_dropout=0.
            )
        )(masked_inputs)
        decoded_seq = RNNDecoder(
            RNNCell(
                LSTM(
                    units=self.latent_size,
                    use_bias=True,
                    implementation=2,
                    dropout=0.,
                    recurrent_dropout=0.,
                ),
                Dense(
                    units=self.encoding_size,
                    activation='tanh'
                ),
                dense_dropout=0.
            )
        )(encoded_seq)
        outputs = TimeDistributed(
            Dense(
                units=self.word_embedding_size,
                activation='tanh'
            )
        )(decoded_seq)

        model = Model(inputs, outputs)
        picked = Pick()(encoded_seq)
        encoder = Model(inputs, picked)

        optimizer = RMSprop(
            lr=self.learning_rate,
            rho=rho,
            decay=decay,
        )
        model.compile(loss='cosine_proximity', optimizer=optimizer)
        return model, encoder

    def load_model(self, file_path):
        self.model = self.load_customed_model(file_path)
        picked = Pick()(self.model.get_layer(index=2).output)
        self.encoder = Model(
            self.model.input,
            picked
        )
        self.word_embedding_size = self.model.input_shape[2]
        self.max_length = self.model.input_shape[1]
        self.latent_size = self.model.get_layer(index=2).layer.recurrent_layer.units
        self.encoding_size = self.model.get_layer(index=3).input_shape[2]

        self.input_transformer = WordEmbeddingTransformer(
            self.word2vec_model, self.max_length
        )
        self.output_transformer = WordEmbeddingTransformer(
            self.word2vec_model, self.max_length
        )
