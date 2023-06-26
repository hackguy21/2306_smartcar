import cv2
import numpy as np 



def lanefilter(img):
    
    # Image Canny 처리
    imgResize = cv2.resize(img, (640,480)) # 웹캠 이미지는 640x480 이나, 다른 사이즈의 사진을 고려해 Resize
    imgGray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY) # Grayscale로 변환. 기존의 이미지가 RGB값의 640*480*3행렬이지만, 이제 640*480행렬 
    imgBlur = cv2.GaussianBlur(imgGray, (7,7), 0) # 가우시안블러. 이미지를 흐릿하게 하여 노이즈를 제거
    imgCanny = cv2.Canny(imgBlur,200,250) # 파라메터를 (img,150,200)으로 하면 edge가 더 줄어든 것을 볼수 있음. 구동환경에 따라 파라메터 변경 필요

    kernel = np.ones( (5,5), np.uint8  )

    # 필터만들기
    lane_filter = np.zeros_like((imgResize)) # 좌상단을 0,0으로, x,y 로 표시
    pt1 = np.array([[200,100],[0,300],[0,480],[640,480],[640,300],[440,100]])
    cv2.fillConvexPoly(lane_filter, pt1, (255,255,255))
    #cv2.imshow('lanefilter image',lane_filter) # 필터이미지를 창으로 띄움

    print(lane_filter.shape)
    
    # cv2.waitKey(0)

    # 만든 필터를 기존 이미지에 마스킹하기
    imgMasked = cv2.bitwise_and(imgResize, lane_filter)
    #cv2.imshow('a',imgMasked)
    #cv2.waitKey(0)
    return imgMasked


path = 'opencv_tutorial/Resources/lena.png'
img = cv2.imread(path)
masked = lanefilter(img)
cv2.imshow('a',masked)
cv2.waitKey(0)