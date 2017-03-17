import pickle

class Dictionary(object):

    def __init__(self, offset=2, dictionary=None):
        self.offset = offset
        self.dictionary = {}
        if dictionary is not None:
            self.dictionary = dictionary

    def __getitem__(self, key):
        if key in self.dictionary:
            return self.dictionary[key] + self.offset
        else:
            return 1
    def size(self):
        return len(self.dictionary) + self.offset

    def load_dict(self, dict_path):
        with open(dict_path, 'rb') as dict_file:
            self.dictionary = pickle.load(dict_file, encoding='utf-8')

    def save_dict(self, dict_path):
        with open(dict_path, 'wb') as dict_file:
            pickle.dump(self.dictionary, dict_file)

