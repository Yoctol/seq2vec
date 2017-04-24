"""Sequence-to-Sequence word2vec."""
import numpy as np

from keras import backend as K
import keras.models
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers.core import Masking, Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers.wrappers import Bidirectional
from keras.models import Model, Input
from keras.regularizers import l2
from yklz import LSTMEncoder, LSTMDecoder
from sklearn.preprocessing import normalize

from .base import BaseSeq2Vec
from .base import TrainableInterfaceMixin
from .base import BaseTransformer
from .util import generate_padding_array

def _create_single_layer_seq2seq_model(
        max_length,
        word_embedding_size,
        latent_size,
        learning_rate,
        rho=0.9,
        decay=0.01,
    ):
    inputs = Input(shape=(max_length, word_embedding_size))
    masked_inputs = Masking(mask_value=0.0)(inputs)
    encoded_seq = Bidirectional(
        LSTMEncoder(
            output_units=latent_size,
            use_bias=True,
            kernel_regularizer=l2(0.001),
            recurrent_regularizer=l2(0.001),
            bias_regularizer=l2(0.001),
            implementation=2,
            output_activation='tanh',
            output_dropout=0.1,
            dropout=0.1,
            recurrent_dropout=0.1,
        )
    )(masked_inputs)
    decoded_seq = LSTMDecoder(
        output_units=word_embedding_size,
        return_sequences=True,
        use_bias=True,
        kernel_regularizer=l2(0.001),
        recurrent_regularizer=l2(0.001),
        bias_regularizer=l2(0.001),
        implementation=2,
        output_activation='tanh',
        output_dropout=0.1,
        dropout=0.1,
        recurrent_dropout=0.1,
    )(encoded_seq)
    output = TimeDistributed(
        Dense(
            word_embedding_size,
            use_bias=True,
            activation='tanh',
            kernel_regularizer=l2(0.001),
            bias_regularizer=l2(0.001),
        )
    )(decoded_seq)
    model = Model(inputs, output)
    encoder = Model(inputs, encoded_seq)

    optimizer = RMSprop(
        lr=learning_rate,
        rho=rho,
        decay=decay,
    )
    model.compile(loss='cosine_proximity', optimizer=optimizer)
    return model, encoder

class Seq2vecWord2vecSeqTransformer(BaseTransformer):

    def __init__(self, word2vec_model, max_length, inverse):
        self.max_length = max_length
        self.word_embedding_size = word2vec_model.get_size()
        self.word2vec = word2vec_model
        self.inverse = inverse

    def seq_transform(self, seq):
        transformed_seq = []
        for word in seq:
            try:
                word_arr = self.word2vec[word]
                normalize(word_arr.reshape(1, -1), copy=False)
                transformed_seq.append(
                    word_arr.reshape(self.word_embedding_size)
                )
            except KeyError:
                pass
        return transformed_seq

    def __call__(self, seqs):
        array = generate_padding_array(
            seqs, self.seq_transform, np.zeros(self.word_embedding_size),
            self.max_length, inverse=self.inverse
        )
        return array

class Seq2SeqWord2Vec(TrainableInterfaceMixin, BaseSeq2Vec):
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
            learning_rate=0.0001,
            latent_size=20,
        ):
        self.word2vec_model = word2vec_model
        self.input_transformer = Seq2vecWord2vecSeqTransformer(
            word2vec_model, max_length, True
        )
        self.output_transformer = Seq2vecWord2vecSeqTransformer(
            word2vec_model, max_length, False
        )
        self.word_embedding_size = word2vec_model.get_size()
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.latent_size = latent_size
        self.encoding_size = latent_size * 2

        model, encoder = _create_single_layer_seq2seq_model(
            max_length=self.max_length,
            word_embedding_size=self.word_embedding_size,
            latent_size=self.latent_size,
            learning_rate=self.learning_rate,
        )
        self.model = model
        self.encoder = encoder

    def transform(self, seqs):
        transformation = super(Seq2SeqWord2Vec, self).transform(seqs)
        return transformation[:, 0, :]

    def load_customed_model(self, file_path):
        return keras.models.load_model(
            file_path, custom_objects={
                'LSTMEncoder':LSTMEncoder, 'LSTMDecoder':LSTMDecoder
            }
        )

    def load_model(self, file_path):
        self.model = self.load_customed_model(file_path)
        self.encoder = Model(
            self.model.input, self.model.get_layer(index=2).output
        )
        self.word_embedding_size = self.model.input_shape[2]
        self.max_length = self.model.input_shape[1]
        self.latent_size = self.model.get_layer(index=3).input_shape[2] // 2
        self.encoding_size = self.latent_size * 2

        self.input_transformer = Seq2vecWord2vecSeqTransformer(
            self.word2vec_model, self.max_length, True
        )
        self.output_transformer = Seq2vecWord2vecSeqTransformer(
            self.word2vec_model, self.max_length, False
        )
