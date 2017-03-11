"""Sequence-to-Sequence Auto Encoder."""
import numpy as np

from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
from keras.layers import Input, LSTM, RepeatVector
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense, Dropout, Activation
from keras.models import Model

from yoctol_utils.hash import consistent_hash

from .base import BaseSeq2Vec
from .base import TrainableInterfaceMixin
import ipdb

def _create_single_layer_seq2seq_model(
        max_length,
        max_index,
        latent_size,
        learning_rate,
        rho=0.9,
        decay=0.01,
    ):

    model = Sequential()
    model.add(Embedding(max_index, 256, input_length=max_length, mask_zero=True, dropout=0.2))
    model.add(LSTM(output_dim=latent_size, return_sequences=False, name='en_LSTM_1', dropout_W=0.2, dropout_U=0.3))
    model.add(RepeatVector(max_length))
    model.add(LSTM(256, return_sequences=True, name='de_LSTM_1', dropout_W=0.2, dropout_U=0.3))
    model.add(TimeDistributed(Dense(max_index)))
    model.add(Dropout(0.2))
    model.add(Activation('softmax'))
    encoder = Model(model.input, model.get_layer('en_LSTM_1').output)

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
            hashed_seq.append(self._hash_seq(seq[::-1]))
        data_pad = pad_sequences(
            hashed_seq,
            maxlen=self.max_length,
            value=0,
        )
        return data_pad

    def _generate_padding_answer_array(self, seqs):
        hashed_seq = []
        for seq in seqs:
            hashed_seq.append(_one_hot_encode_seq(self._hash_seq(seq), self.max_index))
        data_pad = pad_sequences(
            hashed_seq,
            maxlen=self.max_length,
            value=np.zeros(self.max_index + 1),
            padding='post', truncating='post'
        )
        return np.array(data_pad)

    def fit(self, train_seqs, predict_seqs=None, verbose=2, nb_epoch=10, validation_split=0.2):
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
                                train_seqs.append(tr_line.strip().split(' '))
                                test_seqs.append(te_line.strip().split(' '))
                                train_len += 1
                            else:
                                train_array = self._generate_padding_array(train_seqs)
                                test_array = self._generate_padding_answer_array(test_seqs)
                                train_seqs = [tr_line.strip().split(' ')]
                                test_seqs = [te_line.strip().split(' ')]
                                train_len = 1
                                yield (train_array, test_array)
                        train_array = self._generate_padding_array(train_seqs)
                        test_array = self._generate_padding_answer_array(test_seqs)
                        yield (train_array, test_array)
                train_file.close()
                test_file.close()

            else:
                with open(train_file_path, 'r', encoding='utf-8') as train_file:
                    train_seqs = []
                    train_len = 0
                    for line in train_file:
                        if train_len < batch_size:
                            train_seqs.append(line.strip().split(' '))
                            train_len += 1
                        else:
                            train_array = self._generate_padding_array(train_seqs)
                            answer_array = self._generate_padding_answer_array(train_seqs)
                            train_seqs = [line.strip().split(' ')]
                            train_len = 1
                            yield (train_array, answer_array)
                    train_array = self._generate_padding_array(train_seqs)
                    answer_array = self._generate_padding_answer_array(train_seqs)
                    yield (train_array, answer_array)
                train_file.close()

    def fit_file(self, train_file_path, predict_file_path, verbose=1,
                 nb_epoch=10, batch_size=1024, batch_number=1024):
        self.model.fit_generator(
            self.file_generator(
                train_file_path, batch_size=batch_size
            ),
            samples_per_epoch=batch_size * batch_number,
            validation_data=self.file_generator(
                predict_file_path, batch_size=batch_size
            ),
            nb_val_samples=batch_size * batch_number,
            verbose=verbose,
            nb_epoch=nb_epoch,
        )

    def transform(self, seqs):
        test_x = self._generate_padding_array(seqs)
        return self.encoder.predict(test_x)

    def transform_single_sequence(self, seq):
        return self.transform([seq])

    def __call__(self, seqs):
        return self.transform(seqs)
