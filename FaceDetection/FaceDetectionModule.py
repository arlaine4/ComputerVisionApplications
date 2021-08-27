import cv2
import mediapipe as mp
import time

# -------------------------------------------------------#
#     Importing HandTracking Module for test purposes    #
import importlib.util
spec = importlib.util.spec_from_file_location('HandTracking', '../HandTracking/HandTrackingModule.py')
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)

# -------------------------------------------------------#


class FaceDetector:
    def __init__(self, min_detection_confidence=0.7):
        self.min_detection_confidence = min_detection_confidence

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence)

    def find_faces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.res = self.faceDetection.process(imgRGB)
        lst_bbox = []

        if self.res.detections:
            for id, detection in enumerate(self.res.detections):
                bbox = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bbox = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.height * h), int(bbox.width * w)
                lst_bbox.append([id, bbox, detection.score])
                if draw:
                    self.mpDraw.draw_detection(img, detection) #other way to draw bounding box
                    cv2.rectangle(img, bbox, (0, 255, 0), 2)
                    #img = self.fancy_draw(img, bbox) #-> method for fancy corners, not working as I want
                    #cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return lst_bbox, img

    def fancy_draw(self, img, bbox, length=30, t=5, rectangle_thick=1):
        x, y, w, h = bbox
        x1, y1 = y + x, y + h

#        cv2.rectangle(img, bbox, (0, 255, 0), rectangle_thick)
        # Top Left x, y
        cv2.line(img, (x, y), (x + length, y), (0, 255, 0), t)
        cv2.line(img, (x, y), (x, y + length), (0, 255, 0), t)
        # Top Right  x1, y
        cv2.line(img, (x1, y), (x1 - length, y), (0, 255, 0), t)
        cv2.line(img, (x1, y), (x1, y + length), (0, 255, 0), t)
        # Bottom Left x, y1
        cv2.line(img, (x, y1), (x + length, y1), (0, 255, 0), t)
        cv2.line(img, (x, y1), (x, y1 - length), (0, 255, 0), t)
        # Bottom Right x1, y1
        cv2.line(img, (x1, y1), (x1 - length, y1), (0, 255, 0), t)
        cv2.line(img, (x1, y1), (x1, y1 - length), (0, 255, 0), t)

        return img

def main_face_detection():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    detector = FaceDetector()
    hand = foo.HandDetector()
    while True:
        success, img = cap.read()
        # bbox contains the id, bounding box info(xmin, ymin, height, width) and accuracy of the pred
        bbox, img = detector.find_faces(img)
        img = hand.find_hands(img)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(img, f'fps : {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main_face_detection()