import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self,
                static_image_mode=False,
                max_num_hands=2,
                model_complexity = 1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5):

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

    def findAndDrawHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        self.handLandmarks = results.multi_hand_landmarks
        if self.handLandmarks:
            for handLandmark in self.handLandmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmark, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo = 0):
        landmarkList = []

        # if any hands is found
        if self.handLandmarks:
            cHand = self.handLandmarks[handNo]
            for id, lm in enumerate(cHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarkList.append([id, cx, cy])

        return landmarkList

def main():
    videoCapture = cv2.VideoCapture(0)
    handDetector = HandDetector()

    pTime = 0
    cTime = 0
    while True:
        success, img = videoCapture.read()
        img = handDetector.findAndDrawHands(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)

        cv2.imshow("Capture", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()