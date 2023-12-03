import cvlib as cv
import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

sample_num = 0
captured_num = 0

while capture.isOpened():

    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sample_num += 1

    if not ret:
        break;

    faces, confidences = cv.detect_face(frame)

    for idx, f in enumerate(faces):

        startX, startY, endX, endY = f[0], f[1], f[2], f[3]

        if sample_num % 8 == 0:
            captured_num += 1
            face_in_img = frame[startY:endY, startX:endX, :]
            # cv2.imwrite('./data/mask/mask'+str(captured_num)+'.jpg',face_in_img)
            cv2.imwrite('./data/nomask/nomask' + str(captured_num) + '.jpg', face_in_img)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()



