
from math import log, inf
from operator import itemgetter
from transitions import LeftArc, RightArc
from utils.colors import END, BOLD, RED, GREEN, BLUE

class ArcModel():

    def __init__(self, corpus):

        self.corpus = corpus
        self.pos_tags = corpus.get_pos_tags()


    def get_leftarc_probability(self, head, child):
        try:
            return self.probs[LeftArc][(head,child)]
        except KeyError:
            return -inf
        except AttributeError as ex:
            print('Error: Uninitialized Arc Model')
            raise ex


    def get_rightarc_probability(self, head, child):
        try:
            return self.probs[RightArc][(head,child)]
        except KeyError:
            return -inf
        except AttributeError as ex:
            print('Error: Uninitialized Arc Model')
            raise ex


    def train_model(self, debug=False):

        # Initialize probability space as Cartesian product of POS tags
        self.probs = { arc: {} for arc in (LeftArc, RightArc) }
        # Total counts for each Arc, so we can normalize arc counts into probabilities
        sums = { arc: 0 for arc in (LeftArc, RightArc) }

        for sentence in self.corpus:

            # those are ints and denote position in sentence
            for head_index, child_index in sentence.get_gold_arcs():

                # get POS tags
                head = sentence.tokens[head_index].pos
                child = sentence.tokens[child_index].pos

                # Only used for debugging
                headw = sentence.tokens[head_index].form
                childw = sentence.tokens[child_index].form

                # Skip entries with no tag and throw warning
                if head is None or child is None:
                    print('No POS tags for head={} and child={} in [{}]'.format(head_index, child_index, sentence.get_display_sentence() ))
                    continue

                # Index of head is higher than its childs = RightArc
                if head_index < child_index:
                    self.probs[RightArc][(head,child)] = self.probs[RightArc].get( (head,child), 0 ) + 1
                    sums[RightArc] += 1

                    if debug:
                        print('++ RightArc\t[{}{}{}{} ({}{}{}) ==> {}{}{}{} ({}{}{})]\tin [{}]'.format(
                            BOLD, 
                            GREEN, 
                            headw, 
                            END,
                            RED,
                            head,
                            END,
                            BOLD,
                            GREEN,
                            childw,
                            END,
                            RED,
                            child,
                            END,
                            sentence.get_display_sentence() ) )

                # Index of child is higher than its heads = LeftArc
                else:
                    self.probs[LeftArc][(head,child)] = self.probs[LeftArc].get( (head,child), 0 ) + 1
                    sums[LeftArc] += 1

                    if debug:
                        print('++ LeftArc\t[{}{}{}{} ({}{}{}) <== {}{}{}{} ({}{}{})]\tin [{}]'.format(
                            BOLD, 
                            GREEN, 
                            childw, 
                            END,
                            RED,
                            child,
                            END,
                            BOLD,
                            GREEN,
                            headw,
                            END,
                            RED,
                            head,
                            END,
                            sentence.get_display_sentence() ) )

        # Normalize counts into probabilities
        for direction in self.probs:
            for arc in self.probs[direction]:
                try:
                    self.probs[direction][arc] = log(self.probs[direction][arc] / sums[direction] )
                except(ZeroDivisionError, ValueError):
                    self.probs[direction][arc] = -inf

        # We are done
        if not debug:
            print('Built ArcModel with POS-probabilities')
            return

        # But if requested, print collected probabilities in a nice table
        sorted_rightarc = sorted(self.probs[RightArc].items(), key=itemgetter(1), reverse=True)
        sorted_leftarc = sorted(self.probs[LeftArc].items(), key=itemgetter(1), reverse=True)


        print('*' * 125)
        print()
        print(sums)
        print()

        print('*' * 125)
        print()
        print('{}{:>25}\t\t{:>25}{}'.format(BOLD, 'LeftArc', 'RightArc', END))
        print()
        print('*' * 125)
        print()

        for left, right in zip(sorted_leftarc, sorted_rightarc):
            print('{}{}{:<6} <== {:>6}{}  {:>12}\t\t{}{}{:<6} ==> {:>6}{}  {:>12}'.format(
                BOLD,
                BLUE,
                left[0][1],
                left[0][0],
                END,
                round( left[1], 6 ),
                BOLD,
                BLUE,
                right[0][0],
                right[0][1],
                END,
                round( right[1], 6 )
            ))

        print('*' * 125)
        print('{:<20} {}{:>10}\t\t{:>30}{}'.format(
            'Total Arcs', 
            BOLD, 
            sums[LeftArc], 
            sums[RightArc], 
            END ) )
        print('{:<20} {}{:>10}\t\t{:>30}{}'.format(
            'Total POS pairs', 
            BOLD, 
            len(self.probs[LeftArc]),
            len(self.probs[RightArc]),
            END ) )            
        print('{:<20} {}{:>10}\t\t{:>30}{}'.format(
            'Non-zero POS pairs', 
            BOLD, 
            sum(1 if p>0 else 0 for p in self.probs[LeftArc].values()),
            sum(1 if p>0 else 0 for p in self.probs[RightArc].values()),
            END ) )
        print('*' * 125)

        print('Total POS tags: ', len(self.pos_tags))

