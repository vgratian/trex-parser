

## _Trex-Parser_ : transition-based parser for unlabeled dependency parsing

## Introduction

__Trex-Parser__ is a minimalist dependency parser loosely based on the model described in the [2012 paper by Bohner and Nivre](https://www.aclweb.org/anthology/D12-1133/). The feature sets are one-hot vectors of words and POS tags in the stack and buffer. At each time-step, the parser chooses one of the three transitions: _LeftArc_, _RightArc_ or _Shift_ that is assigned the highest joint probability by two models: a Multiclass Perceptron and an Arc model.

The parser is tuned on the English and German datasets of [CoNLL 2006 shared task](https://catalog.ldc.upenn.edu/LDC2015T11).


## Models

The __Arc model__ assigns a probability to the transitions _LeftArc_ and _RightArc_ conditioned on the POS tags of the top 2 elements in the stack. In other words, it measures _P(a|h, c)_: the probability of arc _a_, given head _h_ and child _c_. For convenience we assign probability "1" to the transition _Shift_.

The __multiclass Perceptron__ assigns a probability to each of the three transitions, given the full feature model.


## Feature Model

Features are one-hot representations of the top _w_ elements in stack and buffer. We take the word-form, lemma and POS tags of these elements. The size of _w_ is determined by the hyperparameter _ws_ (window size). The vectors are concatenated, resulting in a high-dimensional feature vector (681,145 and 1,137,613 dimensions for English and German respectively when _w = 12_).

However we only store 3 high-dimensional vectors: the weight vectors of the perceptron. The feature vectors, instead, are the indices of the high-dimensional feature vectors with values _1_. This makes the parser feasibly fast: average inference time per sentence was 0,253s for German and 0.550s (with _b1_ = 25 and _ws_=12).


## Hyper-parameters
- __epochs__ - number of training epochs
- __ws__ - window size, number of top elements in stack and buffer used for feature extraction
- __b1__ - beam size
- __alpha__ - skip arcs that are assigned probabilities less than this value (default 0 means we only allow arcs for POS-pairs that are seen in the traning set)

## TODOs

The model seems to have a bug: it attained UAS 0.935 for English on the development set, but on the test set the unlabeled attachment score was below 0.75.