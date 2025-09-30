import os
import pdb
import numpy as np

file = 'dataset/Nymeria/test.txt'
samples = open(file, 'r').readlines()
np.random.shuffle(samples)
lines = []
for sample in samples[:10]:
    sample = sample.strip('\n')
    text = open(f'dataset/Nymeria/texts/{sample}.txt').readlines()[0]
    action, tokens = text.split('#')[:2]
    lines.append(action +'\n')
with open('assets/nymeria_gt_examples_test.txt', 'w+') as f:
    f.writelines(samples[:10])

with open('assets/nymeria_examples_test.txt', 'w+') as f:
    f.writelines(lines)
pdb.set_trace()