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
            result.append(self.transform(seq))
        return np.array(result)

    @abstractmethod
    def transform(self, seq):
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


class BaseTrainableSeq2Vec(BaseSeq2Vec):
    """Base Trainable sequence-to-vector class."""

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def fit_transform(self):
        pass

    @abstractmethod
    def save(self, path):
        r"""Serialize the transformer.

        Parameters
        ----------
        path: str
            The path to store the Seq2Vec.

        Returns
        -------

        Raises
        ------
        """
        pass


