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
from seq2vec import HashSeq2Vec

transformer = HashSeq2Vec(vector_length=100)
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

* LSTM to LSTM auto-encoder (word embedding)

  ```python
  from seq2vec.word2vec.gensim_word2vec import GensimWord2vec
  from seq2vec.seq2seq_word2vec import Seq2SeqWord2Vec
  
  # load Gensim word2vec from word2vec_model_path
  word2vec = GensimWord2vec(word2vec_model_path)
  
  transformer = Seq2SeqWord2Vec(
        word2vec_model=word2vec,
        max_length=20,
        latent_size=300,
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
  
* CNN to LSTM auto-encoder (word embedding)

  ```python
  from seq2vec.word2vec.gensim_word2vec import GensimWord2vec
  from seq2vec.seq2seq_cnn3D import Seq2SeqCNN
  
  # load Gensim word2vec from word2vec_model_path
  word2vec = GensimWord2vec(word2vec_model_path)
  
  transformer = Seq2SeqCNN(
        word2vec_model=word2vec,
        max_length=20,
        conv_size=5,
        channel_size=10,
        learning_rate=0.05,
        latent_size=300,
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

* CNN to LSTM auto-encoder (char embedding)

  ```python
  from seq2vec.word2vec.gensim_word2vec import GensimWord2vec
  from seq2vec.seq2seq_char2vec import Seq2SeqChar2vec
  
  # load Gensim word2vec from word2vec_model_path
  word2vec = GensimWord2vec(word2vec_model_path)
  
  transformer = Seq2SeqChar2vec(
        word2vec_model=word2vec,
        max_length=20
        embedding_size=100,
        latent_size=300,
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
  
* LSTM to LSTM auto-encoder (char embedding)

 ```python
 from seq2vec import Seq2SeqAutoEncoderUseWordHash

 transformer = Seq2SeqAutoEncoderUseWordHash(
     max_index=1000,
     max_length=10,
     latent_size=20,
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

### Training with file

We provide an example with LSTM to LSTM auto-encoder (word embedding).

Use the following training method while lack of memory is an issue for you.

The file should be a tokenized txt file splitted by whitespace with a sequence
per line.

```python
from seq2vec.word2vec.gensim_word2vec import GensimWord2vec

from seq2vec.seq2seq_word2vec import Seq2SeqWord2Vec
from seq2vec.seq2seq_word2vec import Seq2vecWord2vecSeqTransformer
from seq2vec.data_generator import DataGenterator

word2vec = GensimWord2vec(word2vec_model_path)
max_length = 20
latent_size = 300

transformer = Seq2SeqWord2Vec(
    word2vec_model=word2vec,
    max_length=max_length,
    latent_size=latent_size,
    learning_rate=0.05
)

input_transformer = Seq2vecWord2vecSeqTransformer(
    word2vec, max_length
)
output_transformer = Seq2vecWord2vecSeqTransformer(
    word2vec, max_length
)

train_data = DataGenterator(
    corpus_for_training_path, 
    input_transformer,
    output_transformer, 
    batch_size=128
)
test_data = DataGenterator(
    corpus_for_validation_path, 
    input_transformer,
    output_transformer, 
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

## Lint
```
pylint --rcfile=./yoctol-pylintrc/.pylintrc seq2vec
```


## Test
```
python setup.py test
```

