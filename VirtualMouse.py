import cv2
import numpy as np
import HandDetector
import time

CAPTURE_WIDTH, CAPTURE_HEIGHT = 640, 480

videoCapture = cv2.VideoCapture(0)
videoCapture.set(3, CAPTURE_WIDTH)
videoCapture.set(4, CAPTURE_HEIGHT)

handDetector = HandDetector.HandDetector(max_num_hands=1)

while True:
    success, img = videoCapture.read()
    img = handDetector.findAndDrawHands(img)
    landmarkList = handDetector.findPosition(img)

    if len(landmarkList) != 0:
        x1, y1 = landmarkList[8][1:] # tip of forefinger
        x2, y2 = landmarkList[12][1:] # tip of middle finger

        # print(x1, y1, x2, y2)
        openedFingers = handDetector.getOpenedFinders()

        print(openedFingers)

    cv2.imshow("Virtual Mouse", img)
    cv2.waitKey(1)
