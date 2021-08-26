import time
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prev_time = 0
curr_time = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    res = hands.process(imgRGB)
    if res.multi_hand_landmarks:
        for handLms in res.multi_hand_landmarks:
            for detected_id, lm in enumerate(handLms.landmark):
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                print(detected_id, cx, cy)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, \
                (255, 0, 0), 2)
    cv2.imshow("image", img)
    cv2.waitKey(1)
