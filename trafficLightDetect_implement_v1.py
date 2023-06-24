import numpy as np
import cv2
import Function_Library as fl

def getContours(img):
    # numpy와 cv2를 이용하여 반환값 없이 입력받은 이미지 
    # Canny Image를 넣어주면됨
    #######
    # 0 : 기초 이미지처리

    # imgContour : 박스나, 텍스트를 그려주고 리턴해줄 복사본이미지
    imgContour = img.copy()
    
    # 입력받은 img 행렬을 imgGray - imgBlur - imgCanny 로 처리해줌
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7,7),1)
    imgCanny = cv2.Canny(imgBlur,50,50)

    #######
    # 1 : 닫힌곡선=Contour 추출
    # imgCanny를 cv2라이브러리를 이용해 ontour를 추출
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # 한 이미지 안에서 contour는 여러개가 있을 수 있으며, 위의 코드한줄을 통해 각 contour에 해당하는 정보를 얻었다
    #######
    
    #######
    # 2 : for문을 통해 이미지상에서 감지된 contour들 각각에 대해서 처리를 해줍니다.
    for cnt in contours:
    
        #######
        # 2-1
        # contour의 면적을 구한다.
        area = cv2.contourArea(cnt) #이미지상에 닫힌 곡선의 크기가 클수록 면적도 커진다.
        print(area) 
        #######
        
        #######
        # 2-2
        # 만약 contour area가 특정 임계값을 넘었을 경우 이미지상에 표시해주도록 하자. 
        if area > 500:

            #######
            # 2-3 : Contour의 꼭짓점 개수 구하기
            #######
            # Opencv에서는 RGB값이 아닌 BGR순으로 컬러값의 벡터를 나타냅니다.
            cv2.drawContours(imgContour,cnt,-1,(255,0,0),3) # (255,0,0)은 B가 255값이므로 파랑색이 됩니다.
            peri = cv2.arcLength(cnt,True) # arcLength 라는 함수를 통해서 contour의 길이를 구합니다. 
            #print(peri) # 체크(디버깅)용 코드

            # cv2의 approxPolyDP 함수에서, contour와 arcLength(peri값)을 입력파라메터로 하여 contour의 꼭지점의 개수를 찾아줍니다.
            # 가령 삼각형으로 인식한다면, 3개의 꼭지점을 반환합니다.
            approx = cv2.approxPolyDP(cnt, 0.02*peri,True) 
            #print((approx))# 체크용(디버깅) 코드
            #print(len(approx)) # 체크용(디버깅) 코드
            objCor = len(approx) # objCor 변수에는 삼각형이면 3, 사각형이면 4, 혹은 그 이상은 값들이 들어갑니다.
            
            ############################
            
            #######
            # 2-4 : BoundingBox 그리기
            #######

            # 감지된 물체를 cv2 라이브러리의 boundingRect함수를 이용하여 Bounding Box로 만들어줍니다. 
            # 감지된 다각형의 꼭지점 값을 모은 approx값을 토대로 만들어진 직사각형박스의 좌상단꼭지점의 좌표를 x,y
            # 직사각형의 너비를 w, 높이를 h로 하여 저장합니다.
            x,y,w,h = cv2.boundingRect(approx)
        
            # CV2의 rectange 함수를 이용하여 직사각형을 그려줍니다. 직사각형의 좌상단, 우하단 꼭짓점을 지정
            cv2.rectangle(imgContour, (x,y), (x+w,y+h), (0,255,0)) # BGR의 G 색으로 표현

            #######
            # 2-5 : 텍스트 입력하기
            #######


            # 감지된 contour의 꼭지점 개수에 따라서 이미지 상에 적어줄 문자열을 설정해줍니다.
            if objCor ==3: objectType ="Tri" # 삼각형
            elif objCor == 4: # 사각형
                aspRatio = w/float(h) # bounding box의 너비(w)와 높이(h) 비율을 보고 Square냐 Rectangle이냐 결정
                if aspRatio >0.98 and aspRatio <1.03: objectType= "Square"
                else:objectType="Rectangle"
            elif objCor>4: objectType= "Circles" # 꼭지점이 5이상이면 다 원으로 처리
            else:objectType="None" # 그 이외에는 None 
            # CV2의 putText함수를 이용해 이미지상에 텍스트를 입력
            cv2.putText(imgContour,objectType,
                        (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
                        (0,0,0),2)
    return imgContour
    

# path = "opencv_tutorial\Resources\shapes.png" 
# img = cv2.imread(path) 
# cv2.imshow("abc",getContours(img))
# cv2.waitKey(0)

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

        img = frame0.copy()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (7,7),1)
        imgCanny = cv2.Canny(imgBlur,50,50)


        cv2.imshow("abc",getContours(frame0))
        cv2.imshow("canny",(imgCanny))
        cv2.waitKey(1000)
        #####################################################

        if env.loop_break():
            break
