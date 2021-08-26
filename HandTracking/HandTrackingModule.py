import time
import cv2
import mediapipe as mp

lst_tips = [0, 4, 8, 12, 16, 20]


class HandDetector:
    def __init__(self, mode=False, max_hands=2, min_det_conf=0.75, min_track_conf=0.6):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = min_det_conf
        self.tracking_confidence = min_track_conf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.detection_confidence, self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        global lst_tips
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.res = self.hands.process(imgRGB)
        if self.res.multi_hand_landmarks:
            for handLms in self.res.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    height, width, channel = img.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    if id in lst_tips:
                        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_positions(self, img, handsNo=0, draw=True):
        lst_lms = []
        if self.res.multi_hand_landmarks:
            hand = self.res.multi_hand_landmarks[handsNo]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lst_lms.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        return lst_lms, img

def main_hand_tracking():
    cap = cv2.VideoCapture(0)
    # Timers used for showing fps if show_lms_on_img is set as True
    prev_time = 0
    # List of finger 'tips'
    detector = HandDetector()
    global lst_tips
    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        pos, img = detector.find_positions(img)
        print(pos)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(img, f'fps : {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
        cv2.imshow("image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main_hand_tracking()
