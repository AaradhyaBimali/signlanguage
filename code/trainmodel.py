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
            res=np.load(os.path.join(DATA_PATH, action, str(sequence), f'{frame_num}.npy'))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])