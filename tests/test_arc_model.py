
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..') )
from data_structures.corpus import Corpus
from arc_model import ArcModel


if len(sys.argv) < 2:
    print('Required: corpus filepath')
    sys.exit(0)

corpus = Corpus( sys.argv[1] )

am = ArcModel(corpus)
am.train_model(debug=True)