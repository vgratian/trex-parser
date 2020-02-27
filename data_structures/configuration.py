
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..') )

from transitions import Shift, LeftArc, RightArc
from math import inf

class Configuration():

    def __init__(self, sentence, s=None, b=None, g=None, p=None, h=None, sc=0.0, prob=1.0, f=None):

        self.sentence = sentence
        self.stack = s if s is not None else [0]
        self.buffer = b if b is not None else list(reversed(sentence.get_indices()))
        self.gold_arcs = g if g is not None else sentence.get_gold_arcs()
        self.parsed_arcs = p if p is not None else []
        self.history = h if h is not None else []
        self.score = sc
        self.prob = prob
        self.features = f


    def __str__(self):
        return str(self.is_terminal()) + '\t' + 'STACK=' + str(self.stack) + '\t' + 'BUFFER=' + \
            str(self.buffer) + '\nPARSED ARCS=' + str(self.parsed_arcs) + '\nGOLD ARCS=' + \
            str(self.gold_arcs) + '\nHISTORY=' + str(self.history)
            # str(self.buffer) + '\nHISTORY=' + str(self.history)


    def copy(self):
        return Configuration(   self.sentence, 
                                self.stack.copy(), 
                                self.buffer.copy(), 
                                self.gold_arcs, 
                                self.parsed_arcs.copy(), 
                                self.history.copy(),
                                self.score, 
                                self.prob,
                                None ) # No need to copy features, we are going to overwrite anyway
        

    def is_terminal(self):
        return True if self.stack == [0] and len(self.buffer) == 0 else False


    def get_permissible_transitions(self, arc_model=None, alpha=-inf):

        # Returns list of transition, log prob pairs

        transitions = []

        if len(self.buffer) > 0:
            transitions.append( (Shift, 0) )
        else:
            transitions.append( (Shift, -inf) )

        if len(self.stack) > 1:

            # If we have no Arc Model, return arcs that are technically possible
            if not arc_model:
                transitions.append( (RightArc, 0) )
                transitions.append( (LeftArc, 0) )

            # With Arc Model, return only arcs with probability higher than alpha
            else:
                a, b = self.stack[-2:]
                a_pos = self.sentence.tokens[a].pos
                b_pos = self.sentence.tokens[b].pos

                arc_prob = arc_model.get_rightarc_probability(a_pos, b_pos)
                if arc_prob > alpha:
                    transitions.append( (RightArc, arc_prob) )
                else:
                    transitions.append( (RightArc, -inf) )
                
                arc_prob = arc_model.get_leftarc_probability(b_pos, a_pos)
                if arc_model.get_leftarc_probability(b_pos, a_pos) > alpha:
                    transitions.append( (LeftArc, arc_prob) )
                else:
                    transitions.append( (LeftArc, -inf) )

        else:
            transitions.append( (RightArc, -inf) )
            transitions.append( (LeftArc, -inf) )

        return transitions

