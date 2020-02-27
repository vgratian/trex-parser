
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..') )

from data_structures.sentence import Sentence
from data_structures.token import Token
from transition_extractor import extract_transitions

from random import shuffle


class Corpus():

    def __init__(self, fp):

        """
        Import corpus from file, 
        fp is filepath to file
        """

        self.corpus, self.size = self.import_corpus(fp)


    def import_corpus(self, fp):
        index = 0
        skipped = 0
        corpus = []
        with open(fp) as inp:
            for raw_sentence in inp.read().split('\n\n'):
                index += 1
                try:
                    s = Sentence(raw_sentence)
                except Exception as ex:
                    print('Skipping sentence #{} [{}]: {}'.format(index, raw_sentence, ex))
                    skipped += 1
                else:
                    corpus.append( s )

        size = len(corpus)
        print('Corpus size: {} ({} were skipped)'.format(size, skipped))

        return corpus, size


    def export_corpus(self, fp):

        with open(fp+'_predictions', 'w') as f:
            f.write( '\n\n'.join('\n'.join(str(token) for token in sentence.tokens[1:]) for sentence in self.corpus) )



    def get_dp_labels(self):
        try:
            return self.dp_labels
        except AttributeError:
            self.dp_labels = { token.label for sentence in self.corpus for token in sentence }
            return self.dp_labels
            
    def get_pos_tags(self):
        try:
            return self.pos_tags
        except AttributeError:
            self.pos_tags = { token.pos for sentence in self.corpus for token in sentence }
            return self.pos_tags

    def extract_gold_transitions(self):
        for s in self.corpus:
            s.gold_transitions = extract_transitions(s)

    def shuffle(self):
        shuffle(self.corpus)

    def __iter__(self):
        return iter(self.corpus)