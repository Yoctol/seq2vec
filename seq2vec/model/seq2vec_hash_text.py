"""Hash bag-of-words."""
import numpy as np

from yoctol_utils.hash import consistent_hash

from .seq2vec_base import Seq2VecBase


class Seq2VecHash(Seq2VecBase):
    """Hash words to index.

    Attributes
    ----------
    vector_length: int
        The length of returned vector.
    """

    def __init__(self, vector_length):
        self.vector_length = vector_length

    def transform_single_sequence(self, seq):
        result = np.zeros(self.vector_length)
        for word in seq:
            index = consistent_hash(word) % self.vector_length
            result[index] += 1
        return result

    def transform(self, seqs):
        return self.__call__(seqs)
