# seq2vec
Turn sequence of words into a fix-length representation vector


## Install
```
pip install seq2vec
```
or clone the repo, then
```
python setup.py install
```


## Usage

Simple hash:
```python
from seq2vec import Hash300

words = ['I', 'have', 'an', 'apple']
print(Hash300(ngram=1)(words))
# array([0, 1, 0, ..., 0], dtype=float)
```

TFIDF:
```python

```

Sequence-to-sequence auto-encoder:
```python

```

Seq2Seq
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

## Lint
```
pylint --rcfile=./yoctol-pylintrc/.pylintrc seq2vec
```


## Test
```
python setup.py test
```

