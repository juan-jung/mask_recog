import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_mcs_mouth.xml')

font = ImageFont.truetype('fonts/SCDream6.otf', 20)

while True:    
    ret, frame = capture.read()     
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    faces = face_cascade.detectMultiScale(gray, scaleFactor= 1.5, minNeighbors=3, minSize=(20,20))
    mouths = mouth_cascade.detectMultiScale(gray, scaleFactor= 1.5, minNeighbors=3, minSize=(10,10))
   
    if len(faces) and len(mouths) :
        for  x, y, w, h in faces :
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2, cv2.LINE_4)            
            frame = Image.fromarray(frame)    
            draw = ImageDraw.Draw(frame)       
            draw.text(xy=(x, y-10),  text="No Mask ", font=font, fill=(0,0,255))
            frame = np.array(frame)

    if len(faces) and len(mouths)==0 :
        for  x, y, w, h in faces :
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2, cv2.LINE_4)            
            frame = Image.fromarray(frame)    
            draw = ImageDraw.Draw(frame)       
            draw.text(xy=(x, y-10),  text="Mask On ", font=font, fill=(0,255,0))
            frame = np.array(frame)
    
    cv2.imshow("original", frame)   # frame(카메라 영상)을 original 이라는 창에 띄워줌 
    if cv2.waitKey(1) == ord('q'):  # 키보드의 q 를 누르면 무한루프가 멈춤
            break

capture.release()                   # 캡처 객체를 없애줌
cv2.destroyAllWindows()             # 모든 영상 창을 닫아줌