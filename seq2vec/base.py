"""Base Seq2Vec."""
from abc import abstractmethod


class BaseSeq2Vec(object):
    """Base sequence-to-vector class."""

    @abstractmethod
    def __call__(self, seqs):
        r"""Transform multiple sequences

        Parameters
        ----------
        seqs: list of list of strings
            The

        Returns
        -------
        result_vector : 2-d ndarray


        Raises
        ------
        """
        pass

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
