# Making a Motion detector 

import numpy as np
import cv2

# video = cv2.VideoCapture("cars.mp4")
video = cv2.VideoCapture(0)
# Taking a none frame
frame1 = None
while True:
    # Reading each frame one by one
    ret , frame = video.read()
    # Blurring the frame slightly 
    gray = cv2.GaussianBlur(frame, (21,21), 0)
    # Converting the coloured frame to grayscale or say black and white
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if frame1 is None:
        frame1 = gray
        continue
    # Subtracting both frames
    diff = cv2.absdiff(frame1,gray)
    # Creating a threshold value above this everything will be white and below it everything will be black
    thresh= cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)[1]
    d = cv2.dilate(thresh, None, iterations = 4)
    # Detecting the white pathes
    (contours,_) = cv2.findContours(d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # Making sure that noise is not selected as motion therefore selecting a minimum area only above this
        # motion will be considered rest will be considered as noise
        if cv2.contourArea(c) < 1000:
            continue
        # Creating a boundary around the selected area
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        # Displaying each frame one by one as a video
    cv2.imshow("Detection", frame)
    key = cv2.waitKey(1)
    # if 'q' is pressed the detection will stop
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

    