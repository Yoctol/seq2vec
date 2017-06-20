'''utility functions for data transformation'''
import numpy as np
from sklearn.preprocessing import normalize

from yoctol_utils.hash import consistent_hash

def one_hot_encode_seq(seq, max_index):
    encoded_seq = []
    for index in seq:
        arr = np.zeros(max_index)
        arr[index] = 1
        encoded_seq.append(arr)
    return encoded_seq


def hash_seq(sequence, max_index):
    return [consistent_hash(word) % max_index + 1 for word in sequence]


def word2vec_seq(seq, word2vec):
    transformed_seq = []
    word_embedding_size = word2vec.get_size()
    for word in seq:
        try:
            word_arr = word2vec[word]
            normalize(word_arr.reshape(1, -1), copy=False)
            transformed_seq.append(
                word_arr.reshape(word_embedding_size)
            )
        except KeyError:
            pass
    return transformed_seq
