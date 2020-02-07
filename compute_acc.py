# This is a Python script to compute accuracy numbers of your models on the dev set.
#
# Usage:
#   python compute_acc.py [output file]

import sys

refs = open('dev_ref.txt', 'r').readlines()
outs = open(sys.argv[1], 'r').readlines()

assert len(refs) == len(outs), 'The number of lines should be equal to 643!'

cor = 0
for r, o in zip(refs, outs):
  if r.lower().rstrip() == o.lower().rstrip():
    cor += 1

print('The accuracy number on the dev set is %.2f %%.' % (cor/6.43))
