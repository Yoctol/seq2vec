"""Sequence-to-Sequence word2vec."""
import keras.models
from keras.optimizers import RMSprop
from keras.layers import Input, Conv3D, LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import MaxPooling3D
from keras.models import Model

from yklz import MaskConv, MaskConvNet, MaskPooling, ConvEncoder
from yklz import MaskToSeq, RNNDecoder, RNNCell, Pick
from seq2vec.transformer import WordEmbeddingConv3DTransformer
from seq2vec.transformer import WordEmbeddingTransformer
from seq2vec.model import TrainableSeq2VecBase

class Seq2VecC2RWord(TrainableSeq2VecBase):
    """seq2seq auto-encoder using pretrained word vectors as input
       with ConvNet to RNN architecture.

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
            **kwargs
        ):
        self.word2vec_model = word2vec_model
        self.input_transformer = WordEmbeddingConv3DTransformer(
            word2vec_model,
            max_length,
        )
        self.output_transformer = WordEmbeddingTransformer(
            word2vec_model,
            max_length
        )
        self.word_embedding_size = word2vec_model.get_size()
        self.conv_size = conv_size
        self.channel_size = channel_size
        self.encoding_size = (
            self.word_embedding_size // self.conv_size * self.channel_size
        )

        super(Seq2VecC2RWord, self).__init__(
            max_length,
            latent_size,
            learning_rate
        )
        self.custom_objects['RNNDecoder'] = RNNDecoder
        self.custom_objects['MaskPooling'] = MaskPooling
        self.custom_objects['MaskToSeq'] = MaskToSeq
        self.custom_objects['MaskConv'] = MaskConv
        self.custom_objects['MaskConvNet'] = MaskConvNet
        self.custom_objects['ConvEncoder'] = ConvEncoder
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
                self.max_length,
                self.word_embedding_size,
                1
            )
        )

        masked_inputs = MaskConv(0.0)(inputs)
        masked_seqs = MaskToSeq(
            MaskConv(0.0),
            1
        )(inputs)

        conv = MaskConvNet(
            Conv3D(
                self.channel_size,
                (2, 2, self.conv_size),
                strides=(1, 1, self.conv_size),
                activation='tanh',
                padding='same',
                use_bias=False,
            )
        )(masked_inputs)

        pooling = MaskPooling(
            MaxPooling3D(
                (
                    self.max_length,
                    self.max_length,
                    1
                ),
                padding='same'
            ),
            pool_mode='max'
        )(conv)

        encoded = ConvEncoder()(
            [pooling, masked_seqs]
        )

        decoded = RNNDecoder(
            RNNCell(
                LSTM(
                    self.latent_size,
                    return_sequences=True,
                    implementation=2,
                    unroll=False,
                    dropout=0.,
                    recurrent_dropout=0.,
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
                self.word_embedding_size,
                activation='tanh'
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
        model.compile(loss='cosine_proximity', optimizer=optimizer)
        return model, encoder

    def load_model(self, file_path):
        self.model = self.load_customed_model(file_path)
        picked = Pick()(self.model.get_layer(index=5).output)
        self.encoder = Model(
            self.model.input,
            picked
        )
        self.word_embedding_size = self.model.input_shape[3]
        self.max_length = self.model.input_shape[1]
        self.conv_size = self.model.get_layer(index=2).layer.kernel_size[2]
        self.latent_size = self.model.get_layer(index=6).layer.recurrent_layer.units
        self.channel_size = self.model.get_layer(index=2).layer.filters
        self.encoding_size = self.encoder.output_shape[1]

        self.input_transformer = WordEmbeddingConv3DTransformer(
            self.word2vec_model,
            self.max_length,
        )

        self.output_transformer = WordEmbeddingTransformer(
            self.word2vec_model,
            self.max_length
        )
