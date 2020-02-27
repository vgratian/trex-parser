
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..') )
from data_structures.token import Token


class Sentence():
    def __init__(self, raw_sentence):
        self.tokens = [Token(0, 'ROOT', 'ROOT', 'ROOT')]
        self.tokens += [Token(*token.split('\t')) for token in raw_sentence.split('\n')]
        self.size = len(self.tokens)
        self.gold_transitions = []

    def __str__(self):
        header = '\t'.join(['ID', 'FORM', 'LEMMA', 'POS', 'XPOS', 'MORPH', 'HEAD', 'LABEL', 'LABEL2', 'LABEL3'])
        return header + '\n' + '\n'.join([str(t) for t in self.tokens])


    def get_indices(self):
        try:
            return self.indices
        except AttributeError:
            self.indices = [ t.index for t in self.tokens[1:]]
            return self.indices

    def get_gold_arcs(self):
        try:
            return self.gold_arcs
        except AttributeError:
            self.gold_arcs = [ (t.head, t.index) for t in self.tokens[1:] ]
            return self.gold_arcs


    def get_display_sentence(self):
        return ' '.join([t.form for t in self.tokens[1:]])


    def set_arcs(self, arcs):
        for head, child in arcs:
            self.tokens[child].head = head

    def __iter__(self):
        return iter(self.tokens)