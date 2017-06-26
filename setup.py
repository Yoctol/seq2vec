# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

try:
    long_description = open('README.rst').read()
except IOError:
    long_description = ''

setup(
    name='seq2vec',
    version='0.4.0',
    description='A package to turn sequence of words'
    'into a fix-length representation vector',
    license='GNU 3.0',
    author='cph',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'keras',
        'yoctol_utils',
        'gensim',
        'h5py',
        'yoctol_keras_layer_zoo',
        'tensorflow'
    ],
    long_description=long_description,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
    ]
)
