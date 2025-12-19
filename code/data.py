from function import * #importing all functions from main.py
import cv2

#create directories for each action and store frames
for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)),exist_ok=True)


#initialize mediapipe to detect hands
with mp_hands.Hands(
    model_complexity=0, #chooses less complex model for faster detection
    min_detection_confidence=0.3, #minimum confidence value to detect hand/threshold
    min_tracking_confidence=0.3 #minimum confidence value to track hand
) as hands:
    
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                frame=cv2.imread(f'image/{action}/{sequence}.png') #read image from collected data
            
            if frame is None:
                print("Image not found")
                continue

            image, results = mediapipe_detection(frame, hands) #detect hand landmarks

            #check for hand landmarks in frame
            if results.multi_hand_landmarks:
                print(f'Hand detected for action {action}, sequence {sequence}, frame {frame_num}')
            else:
                print(f'No hand detected for action {action}, sequence {sequence}, frame {frame_num}')

            draw_styles_landmarks(image, results) #draw hand landmarks on image

            #display collection status
            message=(f'Collecting data for action {action}, sequence {sequence}, frame {frame_num}')
            cv2.putText(image, message, (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1,cv2.LINE_AA) #put text on image
            cv2.imshow('OpenCV Feed', image) #show image with landmarks

            #extract and save keypoints
            keypoints=extract_keypoints(results)
            npy_path=os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)


            if cv2.waitKey(10) & 0xFF==ord('q'): #press q to quit
                break

cv2.destroyAllWindows() #close all cv2 windows

            
