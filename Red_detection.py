import cv2
import numpy as np

cap = cv2.VideoCapture(0)


while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.flip(frame,0)
    frame = cv2.flip(frame,1)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if(ret) :
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 빨간색은 hsv에서 0-10, 170-180 두개로 검출해야됨
        lower_red = cv2.inRange(hsv, (0,100,100), (10,255,255))
        upper_red = cv2.inRange(hsv, (170,100,100), (180,255,255))
        added_red = cv2.addWeighted(lower_red, 1.0 ,upper_red, 1.0, 0.0) # 합친거
        red = cv2.bitwise_and(frame, frame, mask= added_red) 


        cv2.imshow('red',red)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 :
            break
        
cap.release()

cv2.destroyAllWindows()        
