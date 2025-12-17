import numpy as np
import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

#image detection function
def mediapipe_detection (image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False #disables writing access to image
    results = model.process(image) #model processes the image
    image.flags.writeable = True #enables writing access to image
    image= cv2.cvtColor(image,cv2,COLOR_RGB2BGR) #converts color to orginal form
    return image, results

#draw the landmarks and hand connections
def draw_landmarks(image, results):
    if results.multi_hand_landmarks: #checks if hands in image
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(), 
                mp_drawing_styles.get_default_hand_connections_style(),
            )

#extract keypoints from the hand landmarks
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        rh=np.array([[res.x,res.y,res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten()
        return rh
    #if landmarks are found will return array
    else:
        return np.zeros(21*3)
    #if no landmarks found will return array of zeros hand has 63 points (21 points with x,y,z)


#define paths and parameters for data detection
DATA_PATH = os.path.join('MP_Data') #folder for collected data
actions = ['A','B','C'] #actions to be detected
no_sequences = 30 #number of sequences for each action
sequence_length = 30 #length of each sequence