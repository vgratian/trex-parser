
from data_structures.configuration import Configuration
from transitions import Shift, LeftArc, RightArc


def extract_transitions(sentence):
    """
    Given an input sentence and gold dependency arcs and labels
    reconstructs the required transitions to parse the sentence

    Input is Sentence object
    """


    config = Configuration(sentence)

    while not config.is_terminal():

        try:
            transition = get_next_transition(config)
        except Exception as ex:
            print(ex)
            return []

        config = transition(config)

    return config.history


def get_next_transition(c):

    if len(c.stack) >= 2:

        s2, s1 = c.stack[-2:]

        if (s1, s2) in c.gold_arcs:
            return LeftArc

        if (s2, s1) in c.gold_arcs:
            permissable = True
            for (h, d) in c.gold_arcs:
                if h == s1 and (s1, d) not in c.parsed_arcs:
                    permissable = False
            if permissable:
                return RightArc

    assert len(c.buffer) >= 1, 'Should return Shift, but Buffer is empty ' \
        '[{}]'.format(c.sentence.get_display_sentence())

    return Shift 

