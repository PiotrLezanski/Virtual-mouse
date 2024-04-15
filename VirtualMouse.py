import cv2
import numpy as np
import HandDetector
import autopy

CAPTURE_WIDTH, CAPTURE_HEIGHT = 640, 480
SCREEN_WIDTH, SCREEN_HEIGHT = autopy.screen.size()
FRAME_REDUCTION = 100

CLICK_DISTANCE = 30  # distance of fingers to threat it as click

# mouse location/smoothening
previousLocX, previousLocY = 0, 0
currentLocX, currentLocY = 0, 0
SMOOTH_CONSTANT = 2

videoCapture = cv2.VideoCapture(0)
videoCapture.set(3, CAPTURE_WIDTH)
videoCapture.set(4, CAPTURE_HEIGHT)

handDetector = HandDetector.HandDetector(max_num_hands=1)

while True:
    success, img = videoCapture.read()
    img = handDetector.findAndDrawHands(img)
    landmarkList = handDetector.findPosition(img)

    if len(landmarkList) != 0:
        x1, y1 = landmarkList[8][1:]  # tip of forefinger
        x2, y2 = landmarkList[12][1:]  # tip of middle finger

        openedFingers = handDetector.getOpenedFinders()

        cv2.rectangle(img, (FRAME_REDUCTION, FRAME_REDUCTION),
                      (CAPTURE_WIDTH - FRAME_REDUCTION, CAPTURE_HEIGHT - FRAME_REDUCTION),
                      (255, 0, 255), 2)

        # if forefinger is up and middle down -> moving mode
        if openedFingers[1] == 1 and openedFingers[2] == 0:
            # convert coordinates from capture to screen
            x3 = np.interp(x1, (FRAME_REDUCTION, CAPTURE_WIDTH - FRAME_REDUCTION), (0, SCREEN_WIDTH))
            y3 = np.interp(y1, (FRAME_REDUCTION, CAPTURE_HEIGHT - FRAME_REDUCTION), (0, SCREEN_HEIGHT))

            # smoothen values
            currentLocX = previousLocX + (x3 - previousLocX) / SMOOTH_CONSTANT
            currentLocY = previousLocY + (y3 - previousLocY) / SMOOTH_CONSTANT

            autopy.mouse.move(SCREEN_WIDTH - currentLocX, currentLocY)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)

            previousLocX, previousLocY = currentLocX, currentLocY

        # if forefinger and middle finger are up -> click
        if openedFingers[1] == 1 and openedFingers[2] == 1:
            distance, centerCords = handDetector.findDistance(8, 12, img)
            if distance < CLICK_DISTANCE:
                cv2.circle(img, (centerCords[0], centerCords[1]),
                           10, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    img = cv2.flip(img, 1)
    cv2.imshow("Virtual Mouse", img)
    cv2.waitKey(1)
