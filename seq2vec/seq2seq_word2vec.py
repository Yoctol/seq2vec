"""Sequence-to-Sequence word2vec."""
import numpy as np

import keras.models
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import LSTM, RepeatVector
from keras.layers.core import Masking
from keras.models import Model
from sklearn.preprocessing import normalize

from .base import BaseSeq2Vec
from .base import TrainableInterfaceMixin
from .base import BaseTransformer
from .util import generate_padding_array

def _create_single_layer_seq2seq_model(
        max_length,
        max_index,
        latent_size,
        learning_rate,
        rho=0.9,
        decay=0.01,
    ):
    model = Sequential()
    model.add(
        Masking(mask_value=0.0, input_shape=(max_length, max_index))
    )
    model.add(
        LSTM(
            latent_size, return_sequences=False, name='en_LSTM_1',
            dropout_W=0.2, dropout_U=0.3
        )
    )
    model.add(RepeatVector(max_length))
    model.add(
        LSTM(
            max_index, return_sequences=True, name='de_LSTM_1',
            dropout_W=0.2, dropout_U=0.3
        )
    )
    encoder = Model(model.input, model.get_layer('en_LSTM_1').output)

    optimizer = RMSprop(
        lr=learning_rate,
        rho=rho,
        decay=decay,
    )
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model, encoder

class Seq2vecWord2vecSeqTransformer(BaseTransformer):

    def __init__(self, word2vec_model, max_length, inverse):
        self.max_length = max_length
        self.max_index = word2vec_model.get_size()
        self.word2vec = word2vec_model
        self.inverse = inverse

    def seq_transform(self, seq):
        transformed_seq = []
        for word in seq:
            try:
                word_arr = self.word2vec[word]
                normalize(word_arr.reshape(1, -1), copy=False)
                transformed_seq.append(word_arr.reshape(self.max_index))
            except KeyError:
                pass
        return transformed_seq

    def __call__(self, seqs):
        array = generate_padding_array(
            seqs, self.seq_transform, np.zeros(self.max_index),
            self.max_length, inverse=self.inverse
        )
        return array

class Seq2SeqWord2Vec(TrainableInterfaceMixin, BaseSeq2Vec):
    """seq2seq auto-encoder using pretrained word vectors as input.

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
            max_length,
            learning_rate=0.0001,
            latent_size=20,
        ):
        self.input_transformer = Seq2vecWord2vecSeqTransformer(
            word2vec_model, max_length, True
        )
        self.output_transformer = Seq2vecWord2vecSeqTransformer(
            word2vec_model, max_length, False
        )
        self.max_index = word2vec_model.get_size()
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

    def fit(self, train_seqs, predict_seqs=None, verbose=2,
            nb_epoch=10, validation_split=0.0):
        train_x = self.input_transformer(train_seqs)
        if predict_seqs is None:
            train_y = self.output_transformer(train_seqs)
        else:
            train_y = self.output_transformer(predict_seqs)

        self.model.fit(
            train_x, train_y,
            verbose=verbose,
            nb_epoch=nb_epoch,
            validation_split=validation_split,
        )

    def fit_generator(self, train_file_generator, test_file_generator,
                      verbose=1, nb_epoch=10, batch_number=1024):
        training_sample_num = train_file_generator.batch_size * batch_number
        testing_sample_num = test_file_generator.batch_size * batch_number
        self.model.fit_generator(
            train_file_generator,
            samples_per_epoch=training_sample_num,
            validation_data=test_file_generator,
            nb_val_samples=testing_sample_num,
            verbose=verbose,
            nb_epoch=nb_epoch,
        )

    def transform(self, seqs):
        test_x = self.input_transformer(seqs)
        return self.encoder.predict(test_x)

    def transform_single_sequence(self, seq):
        return self.transform([seq])

    def __call__(self, seqs):
        return self.transform(seqs)

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = keras.models.load_model(file_path)
        self.encoder = Model(
            self.model.input, self.model.get_layer('en_LSTM_1').output
        )
        self.max_index = self.model.input_shape[2]
        self.max_length = self.model.input_shape[1]
        self.latent_size = self.model.get_layer('en_LSTM_1').output_dim
