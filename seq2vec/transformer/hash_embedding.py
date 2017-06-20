'''hash embedding transformer'''
import numpy as np


from seq2vec.util import generate_padding_array
from .transformer_base import BaseTransformer
from .util import hash_seq, one_hot_encode_seq

class HashIndexTransformer(BaseTransformer):
    '''
        HashIndexTransformer transforms seq into hash index seq
    '''

    def __init__(
            self,
            max_index,
            max_length,
            inverse=False
    ):
        self.max_index = max_index
        self.max_length = max_length
        self.inverse = inverse

    def seq_transform(self, seq):
        return hash_seq(seq, self.max_index - 1)

    def __call__(self, seqs):
        array = generate_padding_array(
            seqs=seqs,
            transfom_function=self.seq_transform,
            default=0,
            max_length=self.max_length,
            inverse=self.inverse
        )
        return array

class OneHotEncodedTransformer(BaseTransformer):
    '''
        OneHotEncodedTransformer transforms seq into one hot encoded seqs
    '''

    def __init__(
            self,
            max_index,
            max_length,
            inverse=False
    ):
        self.max_index = max_index
        self.max_length = max_length
        self.inverse = inverse

    def seq_transform(self, seq):
        transformed_seq = one_hot_encode_seq(
            hash_seq(
                seq,
                self.max_index - 1
            ),
            self.max_index
        )
        return transformed_seq

    def __call__(self, seqs):
        array = generate_padding_array(
            seqs=seqs,
            transfom_function=self.seq_transform,
            default=np.zeros(self.max_index),
            max_length=self.max_length,
            inverse=self.inverse
        )
        return array
