from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

import numpy as np
import os

#map actions to numerical values
label_map=(label:num for num,label in enumerate(actions))

#load and pad sequences
sequences, labels=[], []
for action in actions:
    for sequence in range(no_sequences):
        window=[]
        for frame_num in range(sequence_length):
            npy.path=os.path.join(DATA_PATH, action, str(sequence), f'{frame_num}.npy') #path to each keypoint file

            if os.path.exists(npy_path):
                res=np.load(npy_path,allow_pickle=True) #load keypoints
                window.append(res)
            else:
                window.append(np.zeros(63)) #append zeros if file not found


        sequences.append(window) #append sequence of keypoints
        labels.append(label_map[action]) #append corresponding label