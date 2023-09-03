import cv2, time
import HandTrackinMod as htm
import pyautogui as pg

##########################
wCam , hCam = 1080, 920
cTime = pTime = 0
##########################

cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=2,
                            detectionCon=0.8,
                            trackCon=0.8)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, lmList = detector.findHands(img, imgRGB, draw=False)
    if len(lmList) != 0:
        x1 , y1 = lmList[0][4][1], lmList[0][4][2]
        x2 , y2 = lmList[1][4][1], lmList[1][4][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (x1, y1), 10, (255, 0, 219), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 219), cv2.FILLED)
        cv2.circle(img, (cx, cy), 10, (255, 0, 219), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 219), 4)
        if y1-y2 < -int(img.shape[0] * 0.05): pg.press("d")
        elif y1-y2 > int(img.shape[0] * 0.05): pg.press("a")
        else: print ("Straight")
        p1, p2, p3, p4 = lmList[0][8][2] < lmList[0][5][2],lmList[0][12][2] < lmList[0][9][2],lmList[0][16][2] < lmList[0][13][2],lmList[0][20][2] < lmList[0][17][2]
        q1, q2, q3, q4 = lmList[1][8][2] < lmList[1][5][2],lmList[1][12][2] < lmList[1][9][2],lmList[1][16][2] < lmList[1][13][2],lmList[1][20][2] < lmList[1][17][2]
        if (p1 and p2 and p3 and p4 and q1 and q2 and q3 and q4): pg.press("space")
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (30, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow('Steering using HandGesture', img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cap.release()