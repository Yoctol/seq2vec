"""Sequence-to-Sequence Auto Encoder."""
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
from keras.layers import Input, LSTM, RepeatVector
from keras.layers import Dropout
from keras.models import Model

from yoctol_utils.hash import consistent_hash

from .base import BaseSeq2Vec
from .base import TrainableInterfaceMixin


def _create_single_layer_seq2seq_model(
        max_length,
        max_index,
        latent_size,
        learning_rate,
        rho=0.9,
        decay=0.0,
    ):
    inputs = Input(shape=(max_length, max_index))
    encoded = LSTM(latent_size)(inputs)
    decoded = Dropout(0.3)(encoded)
    decoded = RepeatVector(max_length)(decoded)
    decoded = LSTM(max_index, return_sequences=True)(decoded)
    model = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    optimizer = RMSprop(
        lr=learning_rate,
        rho=rho,
        decay=decay,
    )
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model, encoder


def _one_hot_encode_seq(seq, max_index):
    np_seq = []
    for idx in seq:
        arr = np.zeros(max_index + 1)
        arr[idx] = 1
        np_seq.append(arr)
    return np_seq


class Seq2SeqAutoEncoderUseWordHash(TrainableInterfaceMixin, BaseSeq2Vec):
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
            max_index,
            max_length,
            learning_rate=0.0001,
            latent_size=20,
        ):
        self.max_index = max_index
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.latent_size = latent_size

        model, encoder = _create_single_layer_seq2seq_model(
            max_length=self.max_length,
            max_index=self.max_index + 1,
            latent_size=self.latent_size,
            learning_rate=self.learning_rate,
        )
        self.model = model
        self.encoder = encoder

    def _hash_seq(self, sequence):
        return [consistent_hash(word) % self.max_index + 1 for word in sequence]

    def _generate_padding_array(self, seqs):
        hashed_seq = []
        for seq in seqs:
            hashed_seq.append(self._hash_seq(seq))
        data_pad = pad_sequences(
            hashed_seq,
            maxlen=self.max_length,
            value=0,
        )

        array = []
        for seq in data_pad:
            np_seq = _one_hot_encode_seq(seq, self.max_index)
            array.append(np_seq)
        return np.array(array)

    def fit(self, train_seqs, predict_seqs=None, verbose=2, nb_epoch=10, validation_split=0.0):
        train_x = self._generate_padding_array(train_seqs)
        if predict_seqs is None:
            train_y = train_x
        else:
            train_y = self._generate_padding_array(predict_seqs)
        self.model.fit(
            train_x, train_y,
            verbose=verbose,
            nb_epoch=nb_epoch,
            validation_split=validation_split,
        )

    def transform(self, seqs):
        test_x = self._generate_padding_array(seqs)
        return self.encoder.predict(test_x)

    def transform_single_sequence(self, seq):
        return self.transform([seq])

    def __call__(self, seqs):
        return self.transform(seqs)
