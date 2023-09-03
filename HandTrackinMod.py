import mediapipe as mp
from google.protobuf.json_format import MessageToDict

class handDetector():
    def __init__(self, mode = False, maxHands=2, modelCom=1,detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComp = modelCom
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHand = mp.solutions.hands
        self.hands = self.mpHand.Hands(self.mode, self.maxHands, self.modelComp,self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, img_rgb, draw=True):
        self.results = self.hands.process(img_rgb)
        counter = []
        if self.results.multi_handedness:
            for idx, hand_handedness in enumerate(self.results.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                counter.append(handedness_dict['classification'][0]['index'])
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
                    tipLms.append(lmList)
            else: print("Please provide both hands")
            if counter and tipLms:
                if counter[0] == 1: tipLms[0], tipLms[1] = tipLms[1], tipLms[0]
        return img, tipLms
