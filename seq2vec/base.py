"""Base Seq2Vec."""
from abc import abstractmethod

import numpy as np


class BaseSeq2Vec(object):
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


class TrainableInterfaceMixin(object):
    """Base Trainable sequence-to-vector class."""

    def fit(self, train_seqs, predict_seqs=None, verbose=1,
            epochs=2, validation_split=0.2):
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
        )


    def fit_generator(self, train_file_generator, test_file_generator,
                      verbose=1, epochs=2, batch_number=1024):
        self.model.fit_generator(
            train_file_generator,
            steps_per_epoch=batch_number,
            validation_data=test_file_generator,
            validation_steps=batch_number,
            verbose=verbose,
            epochs=epochs,
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

    def fit_transform(self, seqs):
        self.fit(seqs)
        return self.transform(seqs)

class BaseTransformer(object):
    """
        Base transformer to transform seq to input or output of seq2vec model
    """

    @abstractmethod
    def seq_transform(self, seq):
        pass

    @abstractmethod
    def __call__(self, seqs):
        pass
