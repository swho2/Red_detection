import cv2
import numpy as np
import math


cap = cv2.VideoCapture(0)

class FindNonZero:
    def AllNonZero(height,width,img):
        count_img = 0
        for i in range(height):
            for j in range(width):
                if img[int(i),int(j)] > 0:
                    for find in range (i,height):
                        if img[int(find),int(j)] != 0:
                            count_img += 1
                        else:
                            break
                else:
                    continue
        print(count_img)


while(cap.isOpened()):
    ret, frame = cap.read()
    #frame = cv2.flip(frame,0)
    #frame = cv2.flip(frame,1)
    frame = cv2.resize(frame, dsize=(320, 240), interpolation=cv2.INTER_AREA)
    
    if(ret) :
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = frame.shape[:2] # 높이 너비
        
        # 빨간색은 hsv에서 0-10, 170-180 두개로 검출해야됨
        lower_red = cv2.inRange(hsv, (0,100,100), (5,255,255))
        upper_red = cv2.inRange(hsv, (175,100,100), (180,255,255))
        added_red = cv2.addWeighted(lower_red, 1.0 ,upper_red, 1.0, 0.0) # 합친거
        red = cv2.bitwise_and(frame, frame, mask= added_red)
        h,s,v = cv2.split(red)
        FindNonZero.AllNonZero(height,width,v)
        cv2.imshow('red',red)
        cv2.imshow("new_red",v)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 :
            break
        
cap.release()

cv2.destroyAllWindows()        
