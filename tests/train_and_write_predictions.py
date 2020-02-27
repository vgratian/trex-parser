

import sys
import os
import time

sys.path.insert(1, os.path.join(sys.path[0], '..') )

from data_structures.corpus import Corpus
from perceptron import Perceptron
from arc_model import ArcModel
from utils.colors import END, BOLD, GREEN, RED, BLUE, YELLOW, PINK


if len(sys.argv) < 3:
    print('Required: train and test corpus filepaths')
    sys.exit(0)

# Initialize Train corpus
corpus = Corpus( sys.argv[1] )

# Initialize Test / Dev corpus
test_corpus = Corpus( sys.argv[2] )

# Train Arc model
arcm = ArcModel(corpus)
arcm.train_model()

# Extract gold transitions from gold labels to train perceptron 
corpus.extract_gold_transitions()

# Model parameters
epochs = 75
window_size = 12
alpha = 0
b1 = 25

print(BOLD, RED, 'Training model with: epochs={}, ws={}, b1={}, alpha={}'.format(epochs, window_size, b1, alpha), END)
p = Perceptron(corpus, arc_model=arcm, alpha=alpha)
start = time.time()
p.train(epochs=epochs, b1=b1, debug=False)
end = time.time()
print(GREEN, BOLD, 'Trained Perceptron. Runtime: ', round(end-start, 2), END)


print(PINK, BOLD, 'Testing Model on train dataset [{}]'.format(sys.argv[2]), END)
p.test_corpus = test_corpus
start = time.time()
p.test_model(b1, debug=False, overwrite=True, print_progress=True)
end = time.time()
print(PINK, BOLD, 'Runtime: ', round(end-start, 2), END)

outfp = sys.argv[2].split('/')[-1]
print(YELLOW, BOLD, 'Writing predictions to [{}]'.format(outfp), end=' ')
test_corpus.export_corpus(outfp)
print('DONE', END)

