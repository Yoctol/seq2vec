"""Hash bag-of-words."""
import numpy as np

from .base import BaseSeq2Vec


class HashSeq2Vec(BaseSeq2Vec):
    """Hash words to index.

    Attributes
    ----------
    vector_length: int
        The length of returned vector.
    """

    def __init__(self, vector_length):
        self.vector_length = vector_length

    def transform(self, seq):
        result = np.zeros(self.vector_length)
        for word in seq:
            index = hash(word) % self.vector_length
            result[index] += 1
        return result
