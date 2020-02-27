
import sys
import os
import time

sys.path.insert(1, os.path.join(sys.path[0], '..') )
from data_structures.corpus import Corpus
from transition_extractor import extract_transitions
from utils.colors import END, BOLD, RED, GREEN


if len(sys.argv) < 2:
    print('Required: corpus filepath')
    sys.exit(0)

fp = sys.argv[1]

corpus = Corpus(fp)

start = time.time()

success = 0
failed = 0
    
for s in corpus:

    print(s.get_display_sentence())

    t = extract_transitions(s)

    if t:
        success += 1
        print(BOLD, GREEN, 'FINAL RESULT:', t, END)
    else:
        failed += 1
        print(BOLD, RED, 'FINAL RESULT:', t, END)

end = time.time()

print(BOLD, 'Runtime: ', round(end-start, 2), END)
print('Success: {}, Failed: {} ({}%)'.format(success, failed, round(failed*100/(failed+success), 3)))