from function import *
from tensorflow.keras.models import model_from_json


#load the trained model
json_file=open('model.json','r')
model_json=json_file.read()
json_file.close()

model=model_from_json(model_json)
model.load_weights('model.h5')
print("Model loaded successfully")  # Debug print

#detection and display of variable
sequence=[]
sentence=[]
predictions=[]
accuracy=[]
threshold=0.5  # Lowered threshold for easier detection

cap=cv2.VideoCapture(0)


#initialize mediapipe for hand tracking
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    #loop through every frame
    while cap.isOpened():
        ret, frame=cap.read()

        #process the frame and the region
        cropframe=frame[40:400,0:300]
        frame=cv2.rectangle(frame,(0,40),(300,400),(255,0,0),2)
        image,results=mediapipe_detection(cropframe,hands)

        keypoints=extract_keypoints(results) #extract keypoints using the function
        print(f"Keypoints extracted: {np.sum(keypoints != 0)} non-zero values")  # Debug print
        sequence.append(keypoints) #append keypoints to sequence
        sequence=sequence[-30:] #keep only last 30 frames
        print(f"Sequence length: {len(sequence)}")  # Debug print

        try: 
            if len(sequence)==30:
                res=model.predict(np.expand_dims(sequence, axis=0))[0]#predict on the sequence
                print(f"Predictions: {res}")  # Debug print
                
                detected_action = actions[np.argmax(res)]
                confidence = res[np.argmax(res)]
                print(f"Detected: {detected_action} with confidence {confidence}")  # Debug print
                
                # Always update sentence with current prediction
                sentence.append(detected_action)
                accuracy.append(str(confidence * 100))
                
                # Keep only the last detection
                sentence = sentence[-1:]
                accuracy = accuracy[-1:]


        except Exception as e:
            print(f"Error during prediction: {e}")  # Debug print

        cv2.rectangle(frame,(0,0),(300,40),(245,117,16),-1) #displaying frames
        if sentence:
            output_text = f'Output: {sentence[-1]} ({accuracy[-1]}%)'
        else:
            output_text = 'Output: '
        cv2.putText(frame, output_text, (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', frame)

        if cv2.waitKey(10) & 0xFF==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
