"""Sequence-to-Sequence word2vec."""
import numpy as np
import pdb

from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
from keras.layers import Input, LSTM, RepeatVector
from keras.layers.core import Masking
from keras.layers import Dropout
from keras.models import Model
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize

from yoctol_utils.hash import consistent_hash

from .base import BaseSeq2Vec
from .base import TrainableInterfaceMixin


def _create_single_layer_seq2seq_model(
        max_length,
        max_index,
        latent_size,
        learning_rate,
        rho=0.9,
        decay=0.01,
    ):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(max_length, max_index)))
    model.add(LSTM(latent_size, return_sequences=False, name='en_LSTM_1', dropout_W=0.2, dropout_U=0.3))
    model.add(RepeatVector(max_length))
    model.add(LSTM(max_index, return_sequences=True, name='de_LSTM_1', dropout_W=0.2, dropout_U=0.3))
    encoder = Model(model.input, model.get_layer('en_LSTM_1').output)

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

    def _generate_padding_array(self, seq, inverse=False):
        np_end = np.zeros(self.max_index)

        seq_len = len(seq)
        array_len = 0
        np_seq = []
        for i in range(seq_len):
            try:
                word_arr = self.model_general[seq[i]]
                normalize(word_arr.reshape(1, -1), copy=False)
                np_seq.append(word_arr.reshape(self.max_index))
                array_len += 1
            except KeyError:
                pass
            if array_len == self.max_length:
                break

        end_times = self.max_length - array_len
        for _ in range(end_times):
            np_seq.append(np_end)

        if inverse:
            return np.array(np_seq[::-1])
        else:
            return np.array(np_seq)

    def _generate_padding_seq_array(self, seqs, inverse=False):
        array = []
        for seq in seqs:
            array.append(self._generate_padding_array(seq, inverse))
        return np.array(array)

    def my_generator(self, train_seqs, test_seqs):
        batch_size = 32
        num_seqs = len(train_seqs)
        loop_times = (num_seqs + batch_size - 1) // batch_size
        while True:
            for i in range(loop_times):
                start = i * batch_size
                end = (i + 1) * batch_size
                if end > num_seqs:
                    end = num_seqs

                train_array = self._generate_padding_seq_array(train_seqs[start: end], True)
                test_array = self._generate_padding_seq_array(test_seqs[start: end], False)
                yield (train_array, test_array)

    def fit(self, train_seqs, predict_seqs=None, verbose=2, nb_epoch=10, validation_split=0.0):
        if predict_seqs is None:
            test_seqs = train_seqs
        else:
            test_seqs = predict_seqs
        self.model.fit_generator(
            self.my_generator(train_seqs, test_seqs),
            samples_per_epoch=40000,
            verbose=verbose,
            nb_epoch=nb_epoch,
        )

    def file_generator(self, train_file_path, predict_file_path=None,
                       batch_size=128):
        while True:
            if predict_file_path is not None:
                with open(train_file_path, 'r', encoding='utf-8') as train_file:
                    with open(predict_file_path, 'r', encoding='utf-8') as test_file:
                        train_seqs = []
                        test_seqs = []
                        train_len = 0
                        for tr_line, te_line in zip(train_file, test_file):
                            if train_len < batch_size:
                                train_seqs.append(tr_line.split(' '))
                                test_seqs.append(te_line.split(' '))
                                train_len += 1
                            else:
                                train_array = self._generate_padding_seq_array(train_seqs, True)
                                test_array = self._generate_padding_seq_array(test_seqs, False)
                                train_seqs = [tr_line.split(' ')]
                                test_seqs = [te_line.split(' ')]
                                train_len = 1
                                yield (train_array, test_array)
                        train_array = self._generate_padding_seq_array(train_seqs, True)
                        test_array = self._generate_padding_seq_array(test_seqs, False)
                        yield (train_array, test_array)
                train_file.close()
                test_file.close()

            else:
                with open(train_file_path, 'r', encoding='utf-8') as train_file:
                    train_seqs = []
                    train_len = 0
                    for line in train_file:
                        if train_len < batch_size:
                            train_seqs.append(line.split(' '))
                            train_len += 1
                        else:
                            train_array = self._generate_padding_seq_array(train_seqs, True)
                            answer_array = self._generate_padding_seq_array(train_seqs, False)
                            train_seqs = [line.split(' ')]
                            train_len = 1
                            yield (train_array, answer_array)
                    train_array = self._generate_padding_seq_array(train_seqs, True)
                    answer_array = self._generate_padding_seq_array(train_seqs, False)
                    yield (train_array, answer_array)
                train_file.close()

    def fit_file(self, train_file_path, predict_file_path, verbose=1,
                 nb_epoch=10, batch_size=1024, batch_number=1024):
        self.model.fit_generator(
            self.file_generator(train_file_path, batch_size=batch_size),
            samples_per_epoch=batch_size * batch_number,
            validation_data=self.file_generator(
                predict_file_path, batch_size=batch_size
            ),
            nb_val_samples=batch_size * batch_number,
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
