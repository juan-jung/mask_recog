import cvlib as cv
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image
import datetime

#모델로드
model = load_model('model2.h5')
model.summary()

# open webcam
webcam = cv2.VideoCapture(0)

#폰트 로드
font = ImageFont.truetype('fonts/SCDream6.otf', 20)

#녹화
fourcc = cv2.VideoWriter_fourcc(*'XVID')
is_record = False
on_record = False
cnt_record = 0      # 영상 녹화 시간 관련 변수
max_cnt_record = 5  # 최소 촬영시간

if not webcam.isOpened():
    exit()


while webcam.isOpened():

    # 웹캠에서 frame을 읽어옴, 정상 ret =1. 에러 0q
    ret, frame = webcam.read()

    #시간 출력, 추가기능
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
    nowDatetime_path = now.strftime('%Y-%m-%d %H_%M_%S')

    cv2.rectangle(img=frame, pt1=(10, 15), pt2=(450, 35), color=(0, 0, 0), thickness=-1)

    frame = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame)
    draw.text(xy=(10, 15), text="캡스톤디자인 마스크인식 " + nowDatetime, font=font, fill=(255, 255, 255))
    frame = np.array(frame)

    #에러일 경우
    if not ret:
        exit()

    #face 좌표, confidence 0-1
    face, confidence = cv.detect_face(frame)


    for idx, f in enumerate(face):

        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        face_region = frame[startY:endY, startX:endX]

        face_region1 = cv2.resize(face_region, (224, 224), interpolation=cv2.INTER_AREA)

        x = img_to_array(face_region1)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        prediction = model.predict(x)

        if prediction < 0.5:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            Y = startY - 10 if startY - 10 > 10 else startY + 10
            text = "No Mask ({:.2f}%)".format((1 - prediction[0][0]) * 100)
            cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            #노마스크이면 녹화
            is_record = True  # 녹화 준비
            if on_record == False:
                video = cv2.VideoWriter("./caught/nomask" + nowDatetime_path + ".avi", fourcc, 1,
                                        (frame.shape[1], frame.shape[0]))
            cnt_record = max_cnt_record
            # caught = frame[:, :, :]
            # cv2.imwrite("./caught/" + nowDatetime_path + ".png", caught)

        else:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            Y = startY - 10 if startY - 10 > 10 else startY + 10
            text = "Mask ({:.2f}%)".format(prediction[0][0] * 100)
            cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if is_record == True:
            print('녹화 중')
            video.write(frame)
            cnt_record -= 1
            on_record = True
        if cnt_record == 0:
            is_record = False
            on_record = False
    cv2.imshow("mask recog", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


webcam.release()
cv2.destroyAllWindows()
