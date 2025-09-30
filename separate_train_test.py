import json
import numpy as np
import codecs as cs
import pdb
splits = json.load(open('split.json'))
train_videos = splits['train']
validation_videos = splits['val']
id_list = []
train_list = []
validation_list = []
with cs.open('dataset/Nymeria/train.txt', 'r') as f:
    for line in f.readlines():
        id_name = line
        vid_name = '_'.join(id_name.split('_')[:-2])
        if vid_name.startswith('M'):
            vid_name = vid_name[1:]
        
        if vid_name in train_videos:
            train_list.append(id_name)
        elif vid_name in validation_videos:
            validation_list.append(id_name)
        else:
            continue
open('dataset/Nymeria/train_new.txt', 'w').writelines(train_list)
open('dataset/Nymeria/test_new.txt', 'w').writelines(validation_list)
pdb.set_trace()