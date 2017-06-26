# seq2vec 0.4.0
Turn sequence of words into a fix-length representation vector

This is a version to refactor all the seq2vec structures and use customed layers in yklz.

## Install
```
pip install seq2vec
```
or clone the repo, then install:
```
git clone --recursive https://github.com/Yoctol/seq2vec.git
python setup.py install
```


## Usage

Simple hash:
```python
from seq2vec import Seq2VecHash

transformer = Seq2VecHash(vector_length=100)
seqs = [
    ['我', '有', '一個', '蘋果'],
    ['我', '有', 'pineapple'],
]
result = transformer.transform(seqs)
print(result)
'''
array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
'''
```

Sequence-to-sequence auto-encoder:

* LSTM to LSTM auto-encoder with word embedding (RNN to RNN architecture)

  ```python
  from seq2vec.word2vec import GensimWord2vec
  from seq2vec import Seq2VecR2RWord
  
  # load Gensim word2vec from word2vec_model_path
  word2vec = GensimWord2vec(word2vec_model_path)
  
  transformer = Seq2VecR2RWord(
        word2vec_model=word2vec,
        max_length=20,
        latent_size=300,
        encoding_size=300,
        learning_rate=0.05
  )
  
  train_seq = [
    ['我', '有', '一個', '蘋果'],
    ['我', '有', '筆'],
    ['一個', '鳳梨'],
  ]
  test_seq = [
    ['我', '愛', '吃', '鳳梨'],
  ]
  transformer.fit(train_seq)
  result = transformer.transform(test_seq)
  ```
  
* CNN to LSTM auto-encoder with word embedding (CNN to RNN architecture)

  ```python
  from seq2vec.word2vec import GensimWord2vec
  from seq2vec import Seq2VecC2RWord
  
  # load Gensim word2vec from word2vec_model_path
  word2vec = GensimWord2vec(word2vec_model_path)
  
  transformer = Seq2VecC2RWord(
        word2vec_model=word2vec,
        max_length=20,
        latent_size=300,
        conv_size=5,
        channel_size=10,
        learning_rate=0.05,
  )
  
  train_seq = [
    ['我', '有', '一個', '蘋果'],
    ['我', '有', '筆'],
    ['一個', '鳳梨'],
  ]
  test_seq = [
    ['我', '愛', '吃', '鳳梨'],
  ]
  transformer.fit(train_seq)
  result = transformer.transform(test_seq)
  ```

* CNN to LSTM auto-encoder with char embedding (CNN to RNN architecture)

  ```python
  from seq2vec.word2vec import GensimWord2vec
  from seq2vec import Seq2VecC2RChar
  
  # load Gensim word2vec from word2vec_model_path
  word2vec = GensimWord2vec(word2vec_model_path)
  
  transformer = Seq2VecC2RChar(
        word2vec_model=word2vec,
        max_index=1000,
        max_length=20,
        embedding_size=200,
        latent_size=200,
        learning_rate=0.05,
        channel_size=10,
        conv_size=5
  )
  
  train_seq = [
    ['我', '有', '一個', '蘋果'],
    ['我', '有', '筆'],
    ['一個', '鳳梨'],
  ]
  test_seq = [
    ['我', '愛', '吃', '鳳梨'],
  ]
  transformer.fit(train_seq)
  result = transformer.transform(test_seq)
  ```
  
* LSTM to LSTM auto-encoder with hash word embedding (RNN to RNN architecture)

 ```python
 from seq2vec import Seq2VecR2RHash

 transformer = Seq2VecR2RHash(
     max_index=1000,
     max_length=10,
     latent_size=20,
     embedding_size=200,
     encoding_size=300,
     learning_rate=0.05
 )

 train_seq = [
     ['我', '有', '一個', '蘋果'],
     ['我', '有', '筆'],
     ['一個', '鳳梨'], 
 ]
 test_seq = [
     ['我', '愛', '吃', '鳳梨'],
 ]
 transformer.fit(train_seq)
 result = transformer.transform(test_seq)
 ```

### Training with generator on file

We provide an example with LSTM to LSTM auto-encoder (word embedding).

Use the following training method while lack of memory is an issue for you.

The file should be a tokenized txt file splitted by whitespace with a sequence
per line.

```python
from seq2vec.word2vec import GensimWord2vec

from seq2vec.model import Seq2VecR2RWord
from seq2vec.transformer import WordEmbeddingTransformer
from seq2vec.util import DataGenterator

word2vec = GensimWord2vec(word2vec_model_path)
max_length = 20

transformer = Seq2VecR2RWord(
    word2vec_model=word2vec,
    max_length=max_length,
    latent_size=200,
    encoding_size=300,
    learning_rate=0.05
)

train_data = DataGenterator(
    corpus_for_training_path, 
    transformer.input_transformer,
    transformer.output_transformer, 
    batch_size=128
)
test_data = DataGenterator(
    corpus_for_validation_path, 
    transformer.input_transformer,
    transformer.output_transformer, 
    batch_size=128
)

transformer.fit_generator(
    train_data,
    test_data,
    epochs=10,
    batch_number=1250 # The number of batch per epoch
)

transformer.save_model(model_path) # save your model

# You can reload your model and retrain it.
transformer.load_model(model_path)
transformer.fit_generator(
    train_data,
    test_data,
    epochs=10,
    batch_number=1250 # The number of batch per epoch
)
```

### Customized your seq2vec model with our auto-encoder framework

You can customize your seq2vec model easily with our framework.

```python
import keras
from seq2vec.model import TrainableSeq2VecBase

class YourSeq2Vec(TrainableSeq2VecBase):

   def __init__(self
      max_length,
      latent_size,
      learning_rate
   ):
      # initialize your setting and set input_transformer
      # and output_transformer
      # Input and output transformers transform data from 
      # raw sequence into Keras Layer input format
      # See seq2vec.transformer for more detail

      self.input_transformer = YourInputTransformer()
      self.output_transformer = YourOutputTransformer()

      # add your customized layer
      self.custom_objects = {}
      self.custom_objects[customized_class_name] = customized_class

      super(YourSeq2Vec, self).__init__(
         max_length,
         latent_size,
         learning_rate
      )

   def create_model(self):
      # create and compile your model in this function
      # You should return your model and encoder here
      # encoder is the one encoded input sequences

      model.compile(loss)
      return model, encoder

   def load_model(self, file_path):
      # load your seq2vec model here and set its attribute values
      self.model = self.load_customed_model(file_path)
```

## Lint
```
pylint --rcfile=./yoctol-pylintrc/.pylintrc seq2vec
```


## Test
```
python -m unittest
```

