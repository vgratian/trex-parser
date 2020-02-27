
import sys
import time
import numpy as np
from math import log, exp, inf

from feature_model import FeatureModel
from data_structures.configuration import Configuration
from transitions import Shift, LeftArc, RightArc
from utils.colors import END, BOLD, GREEN, RED, BLUE, YELLOW, PINK
from utils.progress_bar import print_progressbar


class Perceptron():

    def __init__(self, corpus, test_corpus=None, window_size=6, arc_model=None, alpha=0):

        self.corpus = corpus
        self.test_corpus = test_corpus
        self.feature_model = FeatureModel(self.corpus, window_size=window_size)

        # Weights for each transitions
        self.w = {}
        self.w[Shift] = np.zeros( shape=(1, self.feature_model.size) )
        self.w[LeftArc] = np.zeros( shape=(1, self.feature_model.size) )
        self.w[RightArc] = np.zeros( shape=(1, self.feature_model.size) )

        self.arc_model = arc_model
        self.alpha = log(alpha) if alpha else -inf # convert to log probability


    def train(self, epochs, b1, debug=False):

        # Metrics to average weight updates
        self.steps = epochs * self.corpus.size
        self.step = self.steps

        average_uas = 0

        print('*' * 165)
        print('{}{:<5}  {:>12}  {:>10}  {:>14}  {:>14}  {:>14}  {:>7}  {:>12}  {:>14}  {:>14}  {:>14}  {:>7}{}'.format(
            BOLD,
            'Epoch',
            '#Transitions',
            'Accuracy',
            'Shift (p,r)',
            'LeftArc (p,r)',
            'RightArc (p,r)',
            'time',
            'Dev UAS',
            'Shift (p,r)',
            'LeftArc (p,r)',
            'RightArc (p,r)',
            'time',
            END
        ) )
        print('*' * 165)

        for epoch in range(epochs):

            start = time.time()

            self.corpus.shuffle()

            correct = 0
            incorrect = 0

            true_p = { t:0 for t in (Shift, LeftArc, RightArc) }
            false_p = { t:0 for t in (Shift, LeftArc, RightArc) }
            false_n = { t:0 for t in (Shift, LeftArc, RightArc) }

            for s in self.corpus:

                # Skip projective trees, since we don't have gold transitions
                if not s.gold_transitions:
                    print('NO GOLD TRANSITIONS!')
                    continue

                # Construct initial configuration
                c = Configuration(s)
                c.features = self.feature_model.extract(c)

 
                for gold_transition in s.gold_transitions:

                    predictions = []
                    Z = 0

                    for transition, arc_prob in c.get_permissible_transitions():

                        score = float( np.sum( self.w[transition][0, c.features] ) )
                        Z += exp(score)
                        predictions.append( [ transition, 0, score, arc_prob ] )
                    
                    # Normalize transition score into joint probability with arc probability
                    # We use log probabilities!
                    for p in predictions:
                        try:
                            p[1] = log( exp(p[2]) / Z ) + p[3]
                        except(ZeroDivisionError, ValueError):
                            p[1] = -inf
                        

                    predictions.sort(key=lambda x: x[1])
                    predicted = predictions.pop()

                    if predicted[0] != gold_transition:
                        self.update_weights( c.features, predicted[0], gold_transition )
                        incorrect += 1
                        false_p[predicted[0]] += 1
                        false_n[gold_transition] += 1
                    else:
                        correct += 1
                        true_p[predicted[0]] += 1

                    c = gold_transition(c, create_copy=True)
                    c.features = self.feature_model.extract(c)

                self.step -= 1

            # Print some performance metrics for the current epoch
            total_predictions = correct + incorrect
            train_accuracy = round( (correct/total_predictions)*100, 2)
            precision = {t : round( true_p[t] / (true_p[t] + false_p[t]), 2) if (true_p[t] + false_p[t]) != 0 else 0 for t in true_p }
            recall = { t: round( true_p[t] / (true_p[t] + false_n[t]), 2) if (true_p[t] + false_n[t]) != 0 else 0 for t in true_p }
            
            train_epoch_runtime = round( time.time() - start, 2 )

            print('{}{:<5}  {:>12}  {}{:>10}%{}  {:>6} {:>6}  {:>6} {:>6}  {:>6} {:>6}  {:>9}'.format(
                        BOLD,
                        epoch,
                        total_predictions,
                        GREEN,
                        train_accuracy,
                        END,
                        precision[Shift],
                        recall[Shift],
                        precision[LeftArc],
                        recall[LeftArc],
                        precision[RightArc],
                        recall[RightArc],
                        train_epoch_runtime
                ), end='  ' )
            
            if self.test_corpus:
                start = time.time()
                test_accuracy, test_precision, test_recall = self.test_model( b1 , debug)
                test_epoch_runtime = round( time.time() - start, 2 )

                print('{}{}{:>10}%{}    {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}  {:>12}'.format( 
                    GREEN, 
                    BOLD, 
                    test_accuracy, 
                    END,
                    test_precision[Shift],
                    test_recall[Shift],
                    test_precision[LeftArc],
                    test_recall[LeftArc],
                    test_precision[RightArc],
                    test_recall[RightArc],
                    test_epoch_runtime
                    ), end='' )
                average_uas += test_accuracy

            print()

        average_uas /= epochs
        print('*' * 150)
        print('{}{}{:>80} {:>10}%{}'.format(BOLD, BLUE, 'Average UAS:', round(average_uas, 2), END))


    def test_model(self, b1, debug=False, overwrite=False, print_progress=False):

        correct = 0
        incorrect = 0
        true_p = { t:0 for t in (Shift, LeftArc, RightArc) }
        false_p = { t:0 for t in (Shift, LeftArc, RightArc) }
        false_n = { t:0 for t in (Shift, LeftArc, RightArc) }

        if print_progress:
            progress = 0
            print_progressbar(progress, self.test_corpus.size)

        for sentence in self.test_corpus:

            if print_progress:
                progress += 1
                print_progressbar(progress, self.test_corpus.size)

            c0 = Configuration(sentence)
            c0.features = self.feature_model.extract(c0)
            c0.score = 0.0
            beam = [ c0 ]

            if debug:
                print('\n> Testing sentence [{}]'.format(sentence.get_display_sentence()))
                print('> Gold transitions =[{}]\n'.format(' => '.join( transition.__name__ for transition in sentence.gold_transitions ) ) )
                print('*' * 25)

            # Choose and apply best-scoring transitions until we get terminal confiigurations or fail
            while beam and not all( c.is_terminal() for c in beam):

                temp = []

                # First, for each confugration we will calculate probabilities of ALL transitions (including illegal ones)
                # But we will only store the ones that are legal

                for config in beam:

                    # If configuration is already terminal, put it back in the beam
                    if config.is_terminal():
                        temp.append(config)
                        continue

                    predictions = []
                    Z = 0
                    
                    # First iteration to calculate Z, the denominator value of transition probability
                    for transition, arc_prob in config.get_permissible_transitions( self.arc_model, self.alpha ):

                        score = float( np.sum( self.w[transition][0, config.features] ) )
                        Z += exp(score)

                        # Only store transitions with arc probability > 0
                        if arc_prob > self.alpha:
                            new_c = transition(config, create_copy=True)
                            new_c.features = self.feature_model.extract(new_c)
                            predictions.append( (new_c, transition, score, arc_prob) )

                    # Second iteration to normalize transition probabilities and calculate joint probability
                    for new_c, transition, score, arc_prob in predictions:
                        # transition probability
                        try:
                            prob = log( exp(score) / Z )
                        except (ZeroDivisionError, ValueError):
                            prob = -inf
                        # joint probability
                        joint_prob = prob + arc_prob 
                        # local probability (current transition)
                        new_c.score = joint_prob
                        # update global probability (transition sequence)
                        new_c.prob += joint_prob
                        temp.append(new_c)

                        if debug:
                            print('> {} [{}]{}{} => {}{}\t\t score={} prob={} arc_prob={} joint_prob={} global_prob={}'.format(
                                'T' if new_c.is_terminal() else 'N',
                                ' => '.join( transition.__name__ for transition in config.history ),
                                GREEN if all(t1 == t2 for t1, t2 in zip(new_c.history, sentence.gold_transitions) ) else RED,
                                BOLD,
                                transition.__name__,
                                END,
                                round(score, 3),
                                round(prob, 5),
                                round(arc_prob, 5),
                                round(joint_prob, 5),
                                round(new_c.prob, 5)
                            ))
                if debug:
                    print()


                # Sort candidate configurations by probability and prune
                beam = self.prune_beam(temp, b1)

                if debug:
                    if beam:
                        top_scoring = beam[0]
                        print(f'{BOLD}{BLUE}top scoring = {END}{top_scoring.history[-1].__name__} score={round(top_scoring.score,2)}\n')
                        print('*' * 25)
                    else:
                        print(f'{BOLD}{BLUE}top scoring = <beam empty>{END}')

            # After the while loop, if beam is empty we failed to parse the sentence (i.e. no terminal configuration)
            if not beam:
                # This means all arcs were predicted incorrectly
                incorrect += len(sentence.tokens) - 1 # don't count root
                if debug:
                    print()
                    print(RED + ('*' * 25) + END)
                    print('failed to parse any tree')
                    print(RED + ('*' * 25) + END)             
                    yes = input('continue? ')
                continue

            beam.sort(key=lambda x: x.prob, reverse=True)
            winner = beam[0]

            for arc in sentence.get_gold_arcs():
                if arc in winner.parsed_arcs:
                    correct += 1
                else:
                    incorrect += 1
            
            if debug:
                print()
                print(RED + ('*' * 25) + END)
                print('parsed transitions [{}] =\t [{}]'.format(round(winner.score,2), ' => '.join(transition.__name__ for transition in winner.history)))
                print('gold transitions =\t [{}]'.format(' => '.join(transition.__name__ for transition in sentence.gold_transitions)))
                print('gold arcs =', ' '.join('{}{}{}'.format(
                    GREEN if arc in winner.parsed_arcs else RED,
                    str(arc),
                    END ) for arc in sentence.get_gold_arcs()
                    ) )
                print('parsed arcs =', ' '.join('{}{}{}'.format(
                    YELLOW if arc in sentence.get_gold_arcs() else PINK,
                    str(arc),
                    END ) for arc in winner.parsed_arcs
                    ) )
                print(RED + ('*' * 25) + END)
                tmp_accuracy = correct * 100 / (correct + incorrect)
                print(BLUE, 'accuracy = ', GREEN, BOLD, round(tmp_accuracy,2), '%', END)         
                input('enter to continue... ')


            if overwrite:
                sentence.set_arcs(winner.parsed_arcs)


            # This part is independent of what we have done in the beam. We check for each gold configuration, what the best-scoring
            # transition would be, to calculate evaluation metrics 
            h = Configuration(sentence)
            h.features = self.feature_model.extract(h)
            for gold_transition in sentence.gold_transitions:
                predictions = []
                Z = 0
                for transition, arc_prob in h.get_permissible_transitions():
                    score = float( np.sum( self.w[transition][0, h.features] ) )
                    Z += exp(score)
                    predictions.append( [ transition, 0, score, arc_prob ] )
                
                # Normalize transition score into joint probability with arc probability
                for p in predictions:
                    try:
                        p[1] = log( exp(p[2]) / Z ) + p[3]
                    except(ZeroDivisionError, ValueError):
                        p[1] = -inf

                predictions.sort(key=lambda x: x[1])
                predicted = predictions.pop()
                if predicted[0] != gold_transition:
                    self.update_weights( h.features, predicted[0], gold_transition )
                    false_p[predicted[0]] += 1
                    false_n[gold_transition] += 1
                else:
                    true_p[predicted[0]] += 1

                h = gold_transition(h, create_copy=True)
                h.features = self.feature_model.extract(h)

        accuracy = round( correct * 100 / (correct + incorrect), 2 ) if (correct + incorrect) else 0
        precision = { t : round( true_p[t] / (true_p[t] + false_p[t]), 2) if (true_p[t] + false_p[t]) != 0 else 0 for t in true_p }
        recall = { t: round( true_p[t] / (true_p[t] + false_n[t]), 2) if (true_p[t] + false_n[t]) != 0 else 0 for t in true_p }

        return accuracy, precision, recall


    def update_weights(self, features, predicted, gold, learning_rate=0.1):

        self.w[gold][0,features] += learning_rate
        self.w[predicted][0,features] -= learning_rate



    def prune_beam(self, beam, b1):

        # For each transition, choose b1 best scoring ones
        pruned_beam = []

        # first sort configurations by last transition
        sorted_by_transitions = {}
        for c in beam:
            transition = c.history[-1]
            sorted_by_transitions[transition] = sorted_by_transitions.get(transition, []) + [c]

        # sort configurations of each transition by score
        for transition in sorted_by_transitions:
            sorted_by_transitions[transition].sort(key=lambda x: x.prob, reverse=True)
            pruned_beam += sorted_by_transitions[transition][:b1]

        pruned_beam.sort(key=lambda x: x.prob, reverse=True)
        return pruned_beam
