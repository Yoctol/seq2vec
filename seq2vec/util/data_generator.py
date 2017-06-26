

class DataGenterator(object):

    def __init__(
            self, train_file_path, generate_x, generate_y,
            predict_file_path=None, batch_size=128
    ):

        self.train_file_path = train_file_path
        self.predict_file_path = train_file_path
        if predict_file_path is not None:
            self.predict_file_path = predict_file_path

        self.generate_x = generate_x
        self.generate_y = generate_y
        self.batch_size = batch_size

    def array_generator(self, file_path, generating_function, batch_size):
        with open(file_path, 'r', encoding='utf-8') as array_file:
            seqs = []
            seqs_len = 0
            for line in array_file:
                if seqs_len < batch_size:
                    seqs.append(line.strip().split(' '))
                    seqs_len += 1
                else:
                    array = generating_function(seqs)
                    seqs = [line.strip().split(' ')]
                    seqs_len = 1
                    yield array
            array = generating_function(seqs)
            yield array

    def __next__(self):
        while True:
            for x_array, y_array in zip(
                    self.array_generator(
                        self.train_file_path,
                        self.generate_x,
                        self.batch_size
                    ),
                    self.array_generator(
                        self.predict_file_path,
                        self.generate_y,
                        self.batch_size
                    )
            ):
                #assert (len(x_array) == len(y_array)), \
                #    'training data has different length with testing data'
                return (x_array, y_array)
