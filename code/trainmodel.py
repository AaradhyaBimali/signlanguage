from function import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

import numpy as np
import os

#map actions to numerical values
label_map={label:num for num,label in enumerate(actions)}

#load and pad sequences
sequences, labels=[], []
for action in actions:
    for sequence in range(no_sequences):
        window=[]
        for frame_num in range(sequence_length):
            npy_path = os.path.join(DATA_PATH, action, str(sequence), f'{frame_num}.npy') #path to each keypoint file

            if os.path.exists(npy_path):
                res=np.load(npy_path,allow_pickle=True) #load keypoints
                window.append(res)
            else:
                window.append(np.zeros(63)) #append zeros if file not found


        sequences.append(window) #append sequence of keypoints
        labels.append(label_map[action]) #append corresponding label



#prepare the training and validation data
X=np.array(sequences)
y=to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.05,stratify=y) #5% data for testing


#saving the logs of tensorboard
log_dir=os.path.join('logs')
tb_callback=TensorBoard(log_dir=log_dir)

#define a model
model=Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

#compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#train the model
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback], validation_data=(X_test, y_test))
model.summary()

#save the model
model_json=model.to_json()

with open('model.json','w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')
print('Model saved successfully')
