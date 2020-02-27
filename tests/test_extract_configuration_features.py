

import sys
import os
import time

sys.path.insert(1, os.path.join(sys.path[0], '..') )

from data_structures.corpus import Corpus
from feature_model import FeatureModel
from transition_extractor import extract_transitions
from utils.colors import END, BOLD, RED, GREEN


if len(sys.argv) < 2:
    print('Required: corpus filepath')
    sys.exit(0)

fp = sys.argv[1]

corpus = Corpus(fp)

# First we extract gold configurations
start = time.time()
total = 0
failed = 0
for s in corpus:
        t = extract_transitions(s)
        s.gold_transitions = t
        total += 1
        if not t:
            failed += 1
end = time.time()
print(GREEN, BOLD, 'Runtime: ', round(end-start, 2), END)
print('Total: {}, Failed: {} ({}%)'.format(total, failed, round(failed*100/total, 3)))

# Initialize Feature hash tables
start = time.time()
fm = FeatureModel(corpus)
end = time.time()
print(GREEN, BOLD, 'Runtime: ', round(end-start, 2), END)