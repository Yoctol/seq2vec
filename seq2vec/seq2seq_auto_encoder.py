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

def _create_single_layer_seq2seq_model(
        max_length,
        max_index,
        embedding_size,
        latent_size,
        learning_rate,
        rho=0.9,
        decay=0.01,
    ):

    model = Sequential()
    model.add(
        Embedding(
            max_index, embedding_size, input_length=max_length,
            mask_zero=True, dropout=0.2
        )
    )
    model.add(
        LSTM(
            output_dim=latent_size, return_sequences=False,
            name='en_LSTM_1', dropout_W=0.2, dropout_U=0.3
        )
    )
    model.add(
        RepeatVector(max_length)
    )
    model.add(
        LSTM(
            embedding_size, return_sequences=True,
            name='de_LSTM_1', dropout_W=0.2, dropout_U=0.3
        )
    )
    model.add(
        TimeDistributed(Dense(max_index))
    )
    model.add(Dropout(0.2))
    model.add(Activation('softmax'))

    encoder = Model(
        model.input, model.get_layer('en_LSTM_1').output
    )

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

def _hash_seq(sequence, max_index):
    return [consistent_hash(word) % max_index + 1 for word in sequence]

def _generate_padding_array(seqs, max_index, max_length):
    hashed_seq = []
    for seq in seqs:
        hashed_seq.append(_hash_seq(seq[::-1], max_index))
    data_pad = pad_sequences(
        hashed_seq,
        maxlen=max_length,
        value=0,
    )
    return data_pad

def _generate_padding_answer_array(seqs, max_index, max_length):
    hashed_seq = []
    for seq in seqs:
        hashed_seq.append(
            _one_hot_encode_seq(_hash_seq(seq, max_index), max_index)
        )
    data_pad = pad_sequences(
        hashed_seq,
        maxlen=max_length,
        value=np.zeros(max_index + 1),
        padding='post', truncating='post'
    )
    return np.array(data_pad)

class Seq2vecAutoEncoderDataGenterator(object):

    def __init__(
            self, train_file_path, max_index, max_length,
            predict_file_path=None, batch_size=128
    ):

        self.train_file_path = train_file_path
        self.predict_file_path = train_file_path
        if predict_file_path is not None:
            self.predict_file_path = predict_file_path

        self.max_index = max_index
        self.max_length = max_length
        self.batch_size = batch_size

    def array_generator(self, file_path, generating_function, batch_size):
        with open(file_path, 'r', encoding='utf-8') as f:
            seqs = []
            seqs_len = 0
            for line in f:
                if seqs_len < batch_size:
                    seqs.append(line.strip().split(' '))
                    seqs_len += 1
                else:
                    array = generating_function(
                        seqs, self.max_index, self.max_length
                    )
                    seqs = [line.strip().split(' ')]
                    seqs_len = 1
                    yield array
            array = generating_function(
                seqs, self.max_index, self.max_length
            )
            yield array

    def __next__(self):
        while True:
            for x_array, y_array in zip(
                    self.array_generator(
                        self.train_file_path,
                        _generate_padding_array,
                        self.batch_size
                    ),
                    self.array_generator(
                        self.predict_file_path,
                        _generate_padding_answer_array,
                        self.batch_size
                    )
            ):
                assert (len(x_array) == len(y_array)), \
                    'training data has different length with testing data'
                yield (x_array, y_array)

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
            embedding_size=64,
            latent_size=20,
        ):
        self.max_index = max_index
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.latent_size = latent_size

        model, encoder = _create_single_layer_seq2seq_model(
            max_length=self.max_length,
            max_index=self.max_index + 1,
            embedding_size=self.embedding_size,
            latent_size=self.latent_size,
            learning_rate=self.learning_rate,
        )
        self.model = model
        self.encoder = encoder


    def fit(self, train_seqs, predict_seqs=None, verbose=2,
            nb_epoch=10, validation_split=0.2):
        train_x = _generate_padding_array(
            train_seqs, self.max_index, self.max_length
        )
        if predict_seqs is None:
            train_y = _generate_padding_answer_array(
                train_seqs, self.max_index, self.max_length
            )
        else:
            train_y = _generate_padding_answer_array(
                predict_seqs, self.max_index, self.max_length
            )
        self.model.fit(
            train_x, train_y,
            verbose=verbose,
            nb_epoch=nb_epoch,
            validation_split=validation_split,
        )


    def fit_generator(self, train_file_generator, test_file_generator,
                      verbose=1, nb_epoch=10, batch_size=1024,
                      batch_number=1024):
        self.model.fit_generator(
            train_file_generator,
            samples_per_epoch=batch_size * batch_number,
            validation_data=test_file_generator,
            nb_val_samples=batch_size * batch_number,
            verbose=verbose,
            nb_epoch=nb_epoch,
        )

    def transform(self, seqs):
        test_x = _generate_padding_array(seqs, self.max_index, self.max_length)
        return self.encoder.predict(test_x)

    def transform_single_sequence(self, seq):
        return self.transform([seq])

    def __call__(self, seqs):
        return self.transform(seqs)
