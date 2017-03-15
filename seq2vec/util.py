
from keras.preprocessing.sequence import pad_sequences

def generate_padding_array(seqs, transfom_function,
                           default, max_length, inverse=False):
    transformed_seqs = []
    if inverse:
        for seq in seqs:
            transformed_seqs.append(transfom_function(seq[::-1]))
    else:
        for seq in seqs:
            transformed_seqs.append(transfom_function(seq))

    operation = 'post'
    if inverse:
        operation = 'pre'

    data_pad = pad_sequences(
        transformed_seqs,
        maxlen=max_length,
        value=default,
        padding=operation,
        truncating=operation
    )
    return data_pad
