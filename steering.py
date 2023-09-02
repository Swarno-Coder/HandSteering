import cv2
import HandTrackinMod as htm
import numpy as np
import time, math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL


##########################
wCam , hCam = 1080, 920
##########################

cTime = pTime = 0

cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=2,
                            detectionCon=0.8,
                            trackCon=0.8)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img, lmList = detector.findHands(img)
    if len(lmList) != 0:
        x1 , y1 = lmList[0][4][1], lmList[0][4][2]
        x2 , y2 = lmList[1][4][1], lmList[1][4][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (x1, y1), 10, (255, 0, 219), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 219), cv2.FILLED)
        cv2.circle(img, (cx, cy), 10, (255, 0, 219), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 219), 4)
        if y1-y2 < -int(img.shape[0] * 0.05): print ("D")
        elif y1-y2 > int(img.shape[0] * 0.05): print ("A")
        else: print ("Straight")
        '''
        length = math.hypot((x2-x1),(y2-y1))
        volBar = np.interp(length, [48,280], [365, 115])
        volPer = np.interp(length, [48,280], [0, 100])
        print(int(length))
        if length <= 48:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        cv2.rectangle(img, (55, 115), (105, 365), (234,76,97), 3)
        cv2.rectangle(img, (55, int(volBar)), (105, 365), (234,76,97), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)}%', (40, 430), cv2.FONT_HERSHEY_COMPLEX,
                    1.3, (255, 0, 0), 2)
        '''


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (30, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.imshow('Vol Control', img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cap.release()
