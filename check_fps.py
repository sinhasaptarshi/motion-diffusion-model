import numpy as np 
import os
import pdb 

for video in os.listdir('/mnt/Nymeria'):
    if os.path.exists(f'/mnt/Nymeria/{video}/body/xdata.npz'):
        motion = np.load(f'/mnt/Nymeria/{video}/body/xdata.npz')
        print(motion['frameRate'][0])
        if motion['frameRate'][0] != 240:
            pdb.set_trace()