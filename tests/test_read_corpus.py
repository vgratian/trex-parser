
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..') )
from data_structures.corpus import Corpus


if len(sys.argv) < 2:
    print('Required: corpus filepath')
    sys.exit(0)

fp = sys.argv[1]

corpus = Corpus(fp)

length = 0
minn = 99999
maxx = 0
count = 0

for s in corpus:
    length += s.size
    count += 1

    if s.size < minn:
        minn = s.size
    if s.size > maxx:
        maxx = s.size

length /= count

print('Avg sentence length: {}. Min: {}. Max: {}'.format(length, minn, maxx))

answer = input('Print content? [yes/no]: ')
if answer != 'no':
    for sentence in corpus:
        print(sentence)
else:
    sys.exit(0)