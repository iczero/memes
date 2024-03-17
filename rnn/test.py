import sys
from .common import load_dataset

train_set = sys.argv[1]
train_file = open(train_set, 'rb')
train_iter = load_dataset(train_file)

for i in range(1, 4390):
    for j in range(32):
        count, value = next(train_iter)
        if i == 4388:
            print('\n\n\n', count, '\n', repr(value))
