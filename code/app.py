from function import *
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard


#load the trained model
json_file=open('model.json','r')
model_json=json_file.read()
json_file.close()

model=model_from_json(model_json)
model.load_weights('model.h5')

#set colors for different actions
colors=[]
for i in range(0,20):
    colors.append((255,0,0))

#functions for visualizing the probs
def prob_viz(res, actions, input_frame, colors):
    output_frame=input_frame.copy()

    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1) #draw rectangle for each action
        cv2.putText(output_frame, actions[num], (0,85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA) #put probability text

    return output_frame

#detection and display of variable
sequence=[]
sentence=[]
predictions=[]
threshold=0.8

cap=cv2.VideoCapture(0)


#initialize mediapipe for hand tracking
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:


#loop through every frame
while cap.isOpened():
    