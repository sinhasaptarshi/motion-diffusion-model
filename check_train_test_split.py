import numpy as np 
import pdb

train_list = np.load('dataset/t2m_train.npy', allow_pickle=True)[None][0]['name_list']
train_list = ['_'.join(name.split('_')[1:7]) for name in train_list]
test_list = np.load('dataset/t2m_test.npy', allow_pickle=True)[None][0]['name_list']
test_list = ['_'.join(name.split('_')[1:7]) for name in test_list]
pdb.set_trace()
print(len(set(train_list)))
print(len(set(test_list)))
print(len(set(train_list).intersection(set(test_list))))
