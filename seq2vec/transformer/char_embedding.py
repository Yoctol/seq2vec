'''char embedding transformer'''
import numpy as np

from seq2vec.util import generate_padding_array
from .transformer_base import BaseTransformer
from .util import hash_seq, one_hot_encode_seq

class CharEmbeddingOneHotTransformer(BaseTransformer):
    '''
        CharEmbeddingOneHotTransformer one hot encode char in seq into
        3D data format (data_size, timestamps, feature_size)
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
        seq = ''.join(seq)

        return one_hot_encode_seq(
            hash_seq(
                seq,
                self.max_index - 1
            ),
            self.max_index
        )

    def __call__(self, seqs):
        array = generate_padding_array(
            seqs=seqs,
            transfom_function=self.seq_transform,
            default=np.zeros(self.max_index),
            max_length=self.max_length,
            inverse=self.inverse
        )
        return array
