import cv2
import mediapipe as mp
import math


class HandDetector():
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.tipIndexes = [4, 8, 12, 16, 20]
        self.handLandmarks = None
        self.mode = static_image_mode
        self.maxNumberOfHands = max_num_hands
        self.modelComplexity = model_complexity
        self.detectionCon = min_detection_confidence
        self.trackingCon = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,
                                        self.maxNumberOfHands,
                                        self.modelComplexity,
                                        self.detectionCon,
                                        self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findAndDrawHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        self.handLandmarks = results.multi_hand_landmarks
        if self.handLandmarks:
            for handLandmark in self.handLandmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmark, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.landmarkList = []

        # if any hands is found
        if self.handLandmarks:
            cHand = self.handLandmarks[handNo]
            for id, lm in enumerate(cHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmarkList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

        return self.landmarkList

    def getOpenedFinders(self):
        openedFingers = []
        if len(self.landmarkList) != 0:
            # thumb
            if self.landmarkList[self.tipIndexes[0]][1] > self.landmarkList[self.tipIndexes[0] - 1][1]:
                openedFingers.append(1)
            else:
                openedFingers.append(0)

            # rest
            for tipId in range(1, 5):
                if self.landmarkList[self.tipIndexes[tipId]][2] < self.landmarkList[self.tipIndexes[tipId] - 2][2]:
                    openedFingers.append(1)
                else:
                    openedFingers.append(0)

        return openedFingers

    def findDistance(self, tipIndex1, tipIndex2, img=None):
        x1, y1 = self.landmarkList[tipIndex1][1:]
        x2, y2 = self.landmarkList[tipIndex2][1:]
        middleX, middleY = (x1 + x2) // 2, (y1 + y2) // 2

        if img is not None:
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (middleX, middleY), 10, (255, 0, 255), cv2.FILLED)

        return math.hypot(x2 - x1, y2 - y1), [middleX, middleY]
