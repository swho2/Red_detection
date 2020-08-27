import cv2
import numpy as np
import math


cap = cv2.VideoCapture(0)

class FindNonZero:
    def AllNonZero(height,width,img): #빨간색의 물체 크기(총 개수) 찾는 함수
        count_img = 0
        x_axis = np.array([])
        y_axis = np.array([])
        for i in range(width):
            for j in range(height):
                if img[int(j),int(i)] > 0:
                    count_img += 1
                    x_axis = np.append(x_axis,i)
                    y_axis = np.append(y_axis,j)
                else:
                    pass
        print("size: ",count_img)
        print("x_axis: \n", x_axis)
        print("y_axis: \n", y_axis)
        return count_img

    #def FindContour(size,):
    #    if size > 10000:




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
        red = cv2.bitwise_and(frame, frame, mask= added_red) #마스크를 씌움
        h,s,v = cv2.split(red)  #채널 개수를 1개로 만듬
        FindNonZero.AllNonZero(height,width,v)
        cv2.imshow('red',red)
        cv2.imshow("new_red",v)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 :
            break
        
cap.release()

cv2.destroyAllWindows()        
