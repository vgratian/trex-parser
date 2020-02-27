
import sys
import os
import time

sys.path.insert(1, os.path.join(sys.path[0], '..') )

from data_structures.corpus import Corpus
from feature_model import FeatureModel


if len(sys.argv) < 2:
    print('Required: corpus filepath')
    sys.exit(0)

fp = sys.argv[1]

corpus = Corpus(fp)

start = time.time()

fm = FeatureModel(corpus)

end = time.time()

print('Runtime: ', round(end-start, 2))