'''word embedding transformer'''
import numpy as np

from seq2vec.util import generate_padding_array
from .transformer_base import BaseTransformer
from .util import word2vec_seq

class WordEmbeddingTransformer(BaseTransformer):
    '''
        WordEmbedding transformer transforms seq into 3D
        (data_size, timestamps, feature_size) format.
    '''

    def __init__(
            self,
            word2vec_model,
            max_length,
            inverse=False
    ):
        self.max_length = max_length
        self.word_embedding_size = word2vec_model.get_size()
        self.word2vec = word2vec_model
        self.inverse = inverse

    def seq_transform(self, seq):
        return word2vec_seq(seq, self.word2vec)

    def __call__(self, seqs):
        array = generate_padding_array(
            seqs=seqs,
            transfom_function=self.seq_transform,
            default=np.zeros(self.word_embedding_size),
            max_length=self.max_length,
            inverse=self.inverse
        )
        return array

class WordEmbeddingConv3DTransformer(BaseTransformer):

    def __init__(
            self,
            word2vec_model,
            max_length,
    ):
        self.max_length = max_length
        self.embedding_size = word2vec_model.get_size()
        self.word2vec = word2vec_model

    def seq_transform(self, seq):
        transformed_seq = word2vec_seq(
            seq,
            self.word2vec
        )
        transformed_array = np.zeros((
            self.max_length, self.max_length, self.embedding_size
        ))

        seq_length = len(transformed_seq)
        if seq_length > self.max_length:
            seq_length = self.max_length

        for i in range(seq_length):
            for j in range(seq_length):
                if i > j:
                    transformed_array[i, j, :] = transformed_array[j, i, :]
                else:
                    transformed_array[i, j, :] = (
                        transformed_seq[i] + transformed_seq[j]
                    ) / 2

        return transformed_array.reshape(
            self.max_length,
            self.max_length,
            self.embedding_size,
            1
        )

    def __call__(self, seqs):
        array_list = []
        for seq in seqs:
            transformed_array = self.seq_transform(seq)
            array_list.append(transformed_array)
        return np.array(array_list)
