import json 
import numpy as np 
import os 
import pdb
np.random.seed(1234)

root = f'../HumanML3D/joints_data_hot3d/'
train_ratio = 0.7
sequences = np.array([seq.strip('\n') for seq in open(os.path.join('dataset/HOT3D', 'train.txt')).readlines()])
# sequences = np.array([seq for seq in os.listdir(root)])
videos = np.array(['_'.join(seq.split('_')[0:2]) for seq in sequences])
participants = np.array([video[1:].split('_')[0] if video.startswith('M') else video[1:].split('_')[0] for video in videos]) 
train_split = []
test_split = []

for video in np.unique(videos):
    indices = np.where(videos==video)[0]
    selected_sequences = sequences[indices].astype(str)
    np.random.shuffle(selected_sequences)
    train_len = int(train_ratio * len(selected_sequences))
    train_split.extend(list(selected_sequences[:train_len].astype(str)))
    test_split.extend(list(selected_sequences[train_len:].astype(str)))
print(len(train_split))
print(len(test_split))
splits = {}
splits['train'] = train_split
splits['val'] = test_split
with open('HOT3D_splits_per_video.json','w') as f:
    json.dump(splits, f, indent=4)

train_split = []
test_split = []
for participant in np.unique(participants):

    indices = np.where(participants==participant)[0]
    selected_videos = videos[indices].astype(str)
    unique_videos = np.unique(selected_videos)
    selected_sequences = sequences[indices].astype(str)
    train_len = int(train_ratio*len(unique_videos))
    train_videos = unique_videos[:train_len]
    validation_videos = unique_videos[train_len:]
    for i, seq in enumerate(selected_sequences):
        if selected_videos[i] in train_videos:
            train_split.append(selected_sequences[i])
        else:
            test_split.append(selected_sequences[i])
    

print(len(train_split))
print(len(test_split))
splits = {}
splits['train'] = train_split
splits['val'] = test_split
with open('HOT3D_splits_per_participant.json','w') as f:
    json.dump(splits, f, indent=4)
# pdb.set_trace()