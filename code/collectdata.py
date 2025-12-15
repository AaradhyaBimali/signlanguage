import cv2
import os

#vid capture
cap=cv2.videoCapture(0) #0 for default camera


directory='image/' #directory to save images

while True:
    _,frame=cap.read() #read frame from vid capture

    #directory to count number of images in folder
    count={
        'a':len(os.listdir(directory+'A')),
        'b':len(os.listdir(directory+'B')),
        'c':len(os.listdir(directory+'C'))
    }


    #get dimensions of image
    row=frame.shape[1]
    col=frame.shape[0]


    #draw a rectangle to show hand placement
    cv2.rectangle(frame, (0,40), (300,400), (255,255,255),2) #2 is the thickness

    #display the capture region separately
    cv2.imshow('data',frame)
    cv2.imshow('ROI',frame[40:400,0:300]) #region of interest for hand placement

    #crop frame
    frame=frame[40:400,0:300]


    interrupt=cvt.waitKey(10) #wait for key press for 10ms
    if interrupt & 0xFF==ord('a'): #if a is pressed exit loop
        cv2.imwrite(directory+'A/'+str(count['a'])+'.jpg',frame)
        print('A captured')
    if interrupt & 0xFF==ord('b'): #if b is pressed exit loop
        cv2.imwrite(directory+'B/'+str(count['b'])+'.jpg',frame)
        print('B captured')
    if interrupt & 0xFF==ord('c'): #if c is pressed exit loop
        cv2.imwrite(directory+'C/'+str(count['c'])+'.jpg',frame)
        print('C captured')

#release the vid capture device
cap.release()
cv2.destroyAllWindows()
