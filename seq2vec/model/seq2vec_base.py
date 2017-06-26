"""Base Seq2Vec."""
from abc import abstractmethod
import os

import numpy as np
from keras.models import load_model
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint

class Seq2VecBase(object):
    """Base sequence-to-vector class."""

    def __call__(self, seqs):
        r"""Transform multiple sequences

        Parameters
        ----------
        seqs: list of list of strings

        Returns
        -------
        result_vector : 2-d ndarray


        Raises
        ------
        """
        result = []
        for seq in seqs:
            result.append(self.transform_single_sequence(seq))
        return np.array(result)

    @abstractmethod
    def transform_single_sequence(self, seq):
        r"""Transform one sequence

        Parameters
        ----------
        seq: list of strings

        Returns
        -------
        result_vector : 1-d ndarray


        Raises
        ------
        """
        pass


class TrainableSeq2VecBase(Seq2VecBase):
    """Base Trainable sequence-to-vector class."""

    def __init__(
            self,
            max_length=10,
            latent_size=20,
            learning_rate=0.0001,

    ):
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.latent_size = latent_size
        self.custom_objects = {}
        self.model, self.encoder = self.create_model()

        self.best_model_name = self.__class__.__name__ + '_best'
        self.reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            verbose=1,
            factor=0.3,
            patience=3,
            cooldown=3,
            min_lr=1e-6
        )
        self.early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            verbose=1,
        )
        self.model_cp = ModelCheckpoint(
            self.best_model_name,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
        )

    @abstractmethod
    def create_model(self):
        pass

    def fit(
            self,
            train_seqs,
            predict_seqs=None,
            verbose=1,
            epochs=2,
            validation_split=0.2
    ):
        train_x = self.input_transformer(train_seqs)
        if predict_seqs is None:
            train_y = self.output_transformer(train_seqs)
        else:
            train_y = self.output_transformer(predict_seqs)

        self.model.fit(
            train_x, train_y,
            verbose=verbose,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[self.reduce_lr, self.early_stopping, self.model_cp]
        )
        self.model = self.load_customed_model(self.best_model_name)
        os.remove(self.best_model_name)

    def fit_generator(
            self,
            train_file_generator,
            test_file_generator,
            verbose=1,
            epochs=2,
            batch_number=1024
    ):
        self.model.fit_generator(
            train_file_generator,
            steps_per_epoch=batch_number,
            validation_data=test_file_generator,
            validation_steps=batch_number,
            verbose=verbose,
            epochs=epochs,
            callbacks=[self.reduce_lr, self.early_stopping, self.model_cp]
        )
        self.model = self.load_customed_model(self.best_model_name)
        os.remove(self.best_model_name)

    def transform(self, seqs):
        test_x = self.input_transformer(seqs)
        return self.encoder.predict(test_x)

    def transform_single_sequence(self, seq):
        return self.transform([seq])

    def __call__(self, seqs):
        return self.transform(seqs)

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_customed_model(self, file_path):
        return load_model(
            filepath=file_path,
            custom_objects=self.custom_objects
        )

    def fit_transform(self, seqs):
        self.fit(seqs)
        return self.transform(seqs)
