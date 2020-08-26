import cv2
import numpy as np

cap = cv2.VideoCapture(0)
dst = cv2.VideoCapture(0)
lower_blue = (100, 150, 0)
upper_blue = (140, 255, 255)

Threshold = 1
def roi(img, vertices, color3=(255,255,255), color1=255):
    
    mask = np.zeros_like(img)
    
    if len(img.shape) > 2:
        color = color3
    else:
        color = color1
        
    
    cv2.fillPoly(mask, vertices, color)
    
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def draw_lines(img, lines, color=[0, 0, 255], thickness=2): # 선 그리기
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap): # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return lines

def weighted_img(img, initial_img, α=1, β=1., λ=0.): # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, α, img, β, λ)


while(cap.isOpened()):
    ret, frame = cap.read()
    #frame = cv2.flip(frame,0)
   # frame = cv2.flip(frame,1)
    frame = cv2.resize(frame, None, fx=3 / 4, fy=3 / 4, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if(ret) :
        # cv2.imshow('frame',frame)
        blur = cv2.GaussianBlur(gray,(3,3),0)
        canny_cap = cv2.Canny(blur, 70, 210)
        height, width = frame.shape[:2]
        desired_dim = (int(width), int(height))
        vertices = np.array([[(0,0),(0, height), (width,height), (width, 0)]], dtype=np.int32)
        roi_img = roi(canny_cap, vertices, color3=(255,255,255), color1=255)
       

        
        
        # cv2.imshow('frame',frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red = cv2.inRange(hsv, (0,100,100), (10,255,255))
        upper_red = cv2.inRange(hsv, (170,100,100), (180,255,255))
        added_red = cv2.addWeighted(lower_red, 1.0 ,upper_red, 1.0, 0.0)
        red = cv2.bitwise_and(frame, frame, mask= added_red)
        rate = np.count_nonzero(red) / (desired_dim[0] * desired_dim[1])
        
        hough_img = hough_lines(roi_img, 1, 1 * np.pi/180, 30, 10, 20) # 허프 변환
        # result = weighted_img(hough_img, frame)
        line_arr = hough_lines(roi_img, 1, 1 * np.pi/180, 30, 10, 20) # 허프 변환
        line_arr = np.squeeze(line_arr)
            
        # 기울기 구하기
        slope_degree = (np.arctan2(line_arr[:,1] - line_arr[:,3], line_arr[:,0] - line_arr[:,2]) * 180) / np.pi

        # 수평 기울기 제한
        line_arr = line_arr[np.abs(slope_degree)<160]
        slope_degree = slope_degree[np.abs(slope_degree)<160]
        # 수직 기울기 제한
        line_arr = line_arr[np.abs(slope_degree)>95]
        slope_degree = slope_degree[np.abs(slope_degree)>95]
        # 필터링된 직선 버리기
        L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
        temp = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        L_lines, R_lines = L_lines[:,None], R_lines[:,None]
        # 직선 그리기
        draw_lines(temp, L_lines)
        draw_lines(temp, R_lines)

        result = weighted_img(temp, frame) # 원본 이미지에 검출된 선 overlap
        cv2.imshow('result',result) # 결과 이미지 출력
        
        cv2.imshow('red',red)
        if rate > Threshold:
            print("Red!")
            break
        k = cv2.waitKey(1) & 0xFF
        if k == 27 :

            break
        
        
cap.release()

cv2.destroyAllWindows()        
