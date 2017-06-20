import numpy as np

from keras.preprocessing.sequence import pad_sequences

def _padding_array(seqs, from_post, max_length, default):
    seqs_len = len(seqs)
    if seqs_len > max_length:
        if from_post:
            return seqs[0: max_length]
        else:
            start = seqs_len - max_length
            return seqs[start: ]
    elif seqs_len < max_length:
        append_times = max_length - seqs_len
        if from_post:
            list_to_be_append = seqs
        else:
            list_to_be_append = seqs[::-1]

        for _ in range(append_times):
            list_to_be_append.append(default)

        if from_post:
            return list_to_be_append
        else:
            return list_to_be_append[::-1]
    else:
        return seqs

def generate_padding_array(seqs, transfom_function,
                           default, max_length, inverse=False):
    transformed_seqs = []
    if inverse:
        for seq in seqs:
            transformed_seqs.append(
                _padding_array(
                    transfom_function(seq[::-1]), False, max_length, default
                )
            )
    else:
        for seq in seqs:
            transformed_seqs.append(
                _padding_array(
                    transfom_function(seq), True, max_length, default
                )
            )
    return np.array(transformed_seqs)
