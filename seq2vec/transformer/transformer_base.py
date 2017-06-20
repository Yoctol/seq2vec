'''Base transformer class'''
from abc import abstractmethod

class BaseTransformer(object):
    """
        Base transformer transforms seq to the input or output of seq2vec model
    """

    @abstractmethod
    def seq_transform(self, seq):
        pass

    @abstractmethod
    def __call__(self, seqs):
        pass
