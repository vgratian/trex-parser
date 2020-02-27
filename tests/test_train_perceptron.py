
import sys
import os
import time

sys.path.insert(1, os.path.join(sys.path[0], '..') )
from data_structures.corpus import Corpus
from perceptron import Perceptron
from arc_model import ArcModel
from utils.colors import END, BOLD, GREEN, RED, BLUE, YELLOW, PINK


if len(sys.argv) < 2:
    print('Required: corpus filepath')
    sys.exit(0)

# Initialize Train corpus
corpus = Corpus( sys.argv[1] )

# Train Arc model on full data
arcm = ArcModel(corpus)
arcm.train_model()

# Train perceptron on smaller subset
corpus.corpus = corpus.corpus[:1000]
corpus.size = len(corpus.corpus)
print('Stripped train corpus size to 1000')
corpus.extract_gold_transitions()

# If provided, initialize Test / Dev corpus
if len(sys.argv) > 2:
    test_corpus = Corpus( sys.argv[2] )
    test_corpus.corpus = test_corpus.corpus[-50:]
    test_corpus.size = len(test_corpus.corpus)
    print('Stripped test corpus size to 50')
    test_corpus.extract_gold_transitions()
else:
    test_corpus = None

# Model parameters
epochs = 10
window_size = 3
alpha = 0
b1 = 3

print(BOLD, RED, 'Running model with: epochs={}, ws={}, b1={}, alpha={}'.format(epochs, window_size, b1, alpha), END)
p = Perceptron(corpus, test_corpus, arc_model=arcm, alpha=alpha)
start = time.time()
p.train(epochs=epochs, b1=b1, debug=False)
end = time.time()
print(GREEN, BOLD, 'Trained Perceptron. Runtime: ', round(end-start, 2), END)