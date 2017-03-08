"""Sequence-to-Sequence word2vec."""
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
from keras.layers import Input, LSTM, RepeatVector
from keras.layers import Dropout
from keras.models import Model
from gensim.models import Word2Vec

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
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model, encoder

class Seq2SeqWord2Vec(TrainableInterfaceMixin, BaseSeq2Vec):
    """word vector feed to seq2seq auto-encoder.

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
            model_path,
            max_length,
            learning_rate=0.0001,
            latent_size=20,
        ):
        #self.model_yoctol = Word2Vec.load_word2vec_format(
        #    model_path_dict['yoctol'], binary=True
        #)
        self.model_general = Word2Vec.load_word2vec_format(
            model_path, binary=True
        )
        self.max_index = self.model_general.vector_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.latent_size = latent_size

        model, encoder = _create_single_layer_seq2seq_model(
            max_length=self.max_length,
            max_index=self.max_index,
            latent_size=self.latent_size,
            learning_rate=self.learning_rate,
        )
        self.model = model
        self.encoder = encoder

    def _generate_padding_array(self, seq):
        np_end = np.zeros(self.max_index)

        seq_len = len(seq)
        array_len = 0
        np_seq = []
        for i in range(seq_len):
            try:
                word_arr = self.model_general[seq[i]]
                np_seq.append(np.append(word_arr, np.zeros(2)))
                array_len += 1
            except KeyError:
                pass
            if array_len == self.max_length:
                break

        end_times = self.max_length - array_length
        for _ in range(end_times):
            np_seq.append(np_end)
        return np.array(reversed(np_seq))

    def _generate_padding_seq_array(self, seqs):
        array = []
        for seq in seqs:
            array.append(self._generate_padding_array(seq))
        return np.array(array)

    def my_generator(self, train_seqs, test_seqs):
        batch_size = 1024
        num_seqs = len(train_seqs)
        loop_times = (num_seqs + batch_size - 1) // batch_size
        while True:
            for i in range(loop_times):
                start = i * batch_size
                end = (i + 1) * batch_size
                if end > num_seqs:
                    end = num_seqs

                train_array = self._generate_padding_seq_array(train_seqs[start: end])
                test_array = self._generate_padding_seq_array(test_seqs[start: end])
                yield (train_array, test_array)

    def fit(self, train_seqs, predict_seqs=None, verbose=2, nb_epoch=10, validation_split=0.0):
        if predict_seqs is None:
            test_seqs = train_seqs
        else:
            test_seqs = predict_seqs
        self.model.fit_generator(
            self.my_generator(train_seqs, test_seqs),
            samples_per_epoch=10000,
            verbose=verbose,
            nb_epoch=nb_epoch,
        )

    def transform(self, seqs):
        test_x = self._generate_padding_seq_array(seqs)
        return self.encoder.predict(test_x)

    def transform_single_sequence(self, seq):
        return self.transform([seq])

    def __call__(self, seqs):
        return self.transform(seqs)
