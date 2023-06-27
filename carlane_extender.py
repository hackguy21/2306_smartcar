import numpy as np
import cv2
import Function_Library as fl


def lanefilter(img):
    
    # Image Canny 처리
    imgResize = cv2.resize(img, (640,480)) # 웹캠 이미지는 640x480 이나, 다른 사이즈의 사진을 고려해 Resize
    imgGray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY) # Grayscale로 변환. 기존의 이미지가 RGB값의 640*480*3행렬이지만, 이제 640*480행렬 
    imgBlur = cv2.GaussianBlur(imgGray, (7,7), 0) # 가우시안블러. 이미지를 흐릿하게 하여 노이즈를 제거
    imgCanny = cv2.Canny(imgBlur,200,250) # 파라메터를 (img,150,200)으로 하면 edge가 더 줄어든 것을 볼수 있음. 구동환경에 따라 파라메터 변경 필요

    kernel = np.ones( (5,5), np.uint8  )

    # 필터만들기
    lane_filter = np.zeros_like((imgResize)) # 좌상단을 0,0으로, x,y 로 표시
    pt1 = np.array([[0,360],[0,480],[130,480],[130,360]])
    pt2 = np.array([[640,480],[510,480],[510,360],[640,360]])
    cv2.fillConvexPoly(lane_filter, pt1, (255,255,255))
    cv2.fillConvexPoly(lane_filter, pt2, (255,255,255))
    # cv2.imshow('lanefilter image',lane_filter) # 필터이미지를 창으로 띄움

    #print(lane_filter.shape)
    
    # cv2.waitKey(0)

    # 만든 필터를 기존 이미지에 마스킹하기
    imgMasked = cv2.bitwise_and(imgResize, lane_filter)
    #cv2.imshow('a',imgMasked)
    #cv2.waitKey(0)
    return imgMasked




def carlane_extender(img):
    # numpy와 cv2를 이용하여 반환값 없이 입력받은 이미지 
    # Canny Image를 넣어주면됨
    #######
    # 0 : 기초 이미지처리
    state = 'nothing'
    # imgContour : 박스나, 텍스트를 그려주고 리턴해줄 복사본이미지
    imgContour = img.copy()



    # # 입력받은 img 행렬을 imgGray - imgBlur - imgCanny 로 처리해줌
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7,7),1)
    imgCanny = cv2.Canny(imgBlur,50,50)

    edges = cv2.Canny(imgCanny, 50, 150, apertureSize=3)

    # Perform Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    # Draw the lines on the original image
    
    carlaneline = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            slope_left = []
            slope_right = []

            # Check if the line is vertical
            if x2 - x1 != 0:
                # Calculate the slope and y-intercept of the line
                slope = (y2 - y1) / (x2 - x1)
                y_intercept = y1 - slope * x1
                if slope == 0:
                    continue
                if -0.3 < slope < 0.3:
                    continue
                
                if 520 > x1 > 500 or 520 > x2 > 500:
                    continue
                if 140 > x1 > 120 or 140 > x2 > 120:
                    continue

                # Extend the line to the image edges
                y1_extended = 0
                x1_extended = int((y1_extended - y_intercept) / slope)
                y2_extended = imgContour.shape[0]
                x2_extended = int((y2_extended - y_intercept) / slope)

                # Draw the extended line on the image
                cv2.line(imgContour, (x1_extended, y1_extended), (x2_extended, y2_extended), (0, 0, 255), 2)
                if x1>520:
                    slope_right.append(slope)
                elif x1<120:
                    slope_left.append(slope)
                    
        if len(slope_left)!=0:
            print(f"slope_left average is {sum(slope_left)/len(slope_left)}")
        if len(slope_right)!=0:
            print(f"slope_right average is {sum(slope_right)/len(slope_right)}")
                
                

    # Display the image with extended lines
    #cv2.imshow('Extended Lines', imgContour)
    
    return imgContour



    

#######################################33

EPOCH = 500000

if __name__ == "__main__":
    # Exercise Environment Setting
    env = fl.libCAMERA()

    """ Exercise 1: RGB Color Value Extracting """
    ############## YOU MUST EDIT ONLY HERE ##############
    # example = env.file_read("2306_smartcar\Example Image.jpg")
    # R, G, B = env.extract_rgb(example, print_enable=True)
    # quit()
    #####################################################

    # Camera Initial Setting
    ch0, ch1 = env.initial_setting(capnum=2)

    # Camera Reading..
    for i in range(EPOCH):
        _, frame0, _, frame1 = env.camera_read(ch0, ch1)


        cv2.imshow('original',frame0)
        img = frame0.copy()
        masked = lanefilter(img)
        cv2.imshow('masked image',masked)
        houghed = carlane_extender(masked)
        cv2.imshow('masked image', houghed)
        
        
        cv2.waitKey(1000)
        #####################################################

        if env.loop_break():
            break
