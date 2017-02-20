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


## Lint
```
pylint --rcfile=./yoctol-pylintrc/.pylintrc seq2vec
```


## Test
```
python setup.py test
```

