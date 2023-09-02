import cv2
import mediapipe as mp
import time
from google.protobuf.json_format import MessageToDict

class handDetector():
    def __init__(self, mode = False, maxHands=2, modelCom=1,detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComp = modelCom
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHand = mp.solutions.hands
        self.hands = self.mpHand.Hands(self.mode, self.maxHands, self.modelComp,
                                       self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_handedness:
            for idx, hand_handedness in enumerate(self.results.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                whichHand = (handedness_dict['classification'][0]['label'])
                # print(whichHand)
                if whichHand == "Left":
                    RightHand = True
                else:
                    RightHand = False
        tipLms = []
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks)>=2:
                h, w, c = img.shape
                for handLms in self.results.multi_hand_landmarks:
                    if draw: self.mpDraw.draw_landmarks(img, handLms,
                                                self.mpHand.HAND_CONNECTIONS,
                                                connection_drawing_spec=
                                                self.mpDraw.DrawingSpec((233,43,5),
                                                                        thickness=3))
                    lmList = []
                    for id, lm in enumerate(handLms.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])
                    #tipX = handLms.landmark[self.mpHand.HandLandmark.THUMB_TIP].x * w
                    #tipY = handLms.landmark[self.mpHand.HandLandmark.THUMB_TIP].y * h
                    tipLms.append(lmList)
            else: print("Please provide both hands")
        return img, tipLms

def main():
    cTime = pTime = 0

    ##########################
    wCam, hCam = 1080, 920
    ##########################

    cap = cv2.VideoCapture(1)
    cap.set(3, wCam)
    cap.set(4, hCam)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img, lm = detector.findHands(img)
        print(lm)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 3)
        cv2.imshow("Result", img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    main()