"""Base Word2vec class"""
from abc import abstractmethod

class BaseWord2vecClass(object):

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def get_size(self):
        pass
