'''DSSM'''
import numpy as np
import keras
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation
from yoctol_utils.hash import consistent_hash

from seq2vec.base import BaseSeq2Vec, TrainableInterfaceMixin
from seq2vec.base import BaseTransformer

def _create_dssm_model(
        max_index,
        embedding_size,
        latent_size,
        learning_rate,
        rho=0.9,
        decay=0.01,
    ):

    model = Sequential()
    model.add(
        Dense(
            embedding_size, input_shape=(max_index,),
            name='embedding', activation='relu'
        )
    )
    model.add(Dropout(0.2))
    model.add(Dense(latent_size, name='encoder', activation='relu'))
    model.add(Dense(embedding_size, name='decoder', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(max_index, name='output', activation='softmax'))

    encoder = Model(
        model.input, model.get_layer('encoder').output
    )

    optimizer = RMSprop(
        lr=learning_rate,
        rho=rho,
        decay=decay,
    )
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model, encoder

class Seq2vecDSSMTransformer(BaseTransformer):

    def __init__(self, max_index):
        self.max_index = max_index

    def seq_transform(self, seq):
        array = np.zeros(self.max_index)
        for word in seq:
            index = consistent_hash(word) % self.max_index
            array[index] += 1.0
        return array

    def __call__(self, seqs):
        array_list = []
        for seq in seqs:
            array_list.append(self.seq_transform(seq))
        return np.array(array_list)

class Seq2SeqDSSM(TrainableInterfaceMixin, BaseSeq2Vec):
    """Hash words and feed to seq2seq dssm.

    Attributes
    ----------
    max_index: int
        The length of input vector.

    latent_size: int
        The returned latent vector size after encoding.

    """

    def __init__(
            self,
            max_index,
            learning_rate=0.0001,
            embedding_size=64,
            latent_size=20,
        ):
        self.max_index = max_index
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.latent_size = latent_size

        self.input_transformer = Seq2vecDSSMTransformer(
            max_index
        )
        self.output_transformer = Seq2vecDSSMTransformer(
            max_index
        )

        model, encoder = _create_dssm_model(
            max_index=self.max_index,
            embedding_size=self.embedding_size,
            latent_size=self.latent_size,
            learning_rate=self.learning_rate,
        )
        self.model = model
        self.encoder = encoder

    def load_model(self, file_path):
        self.model = keras.models.load_model(file_path)
        self.encoder = Model(
            self.model.input, self.model.get_layer('encoder').output
        )
        self.max_index = self.model.get_layer('embedding').input_dim
        self.embedding_size = self.model.get_layer('embedding').output_dim
        self.latent_size = self.model.get_layer('encoder').output_dim
