import cv2
import mediapipe as mp
import time

# -------------------------------------------------------#
#     Importing HandTracking Module for test purposes    #
import importlib.util
spec = importlib.util.spec_from_file_location('HandTracking', '../HandTracking/HandTrackingModule.py')
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
#                                                        #
# -------------------------------------------------------#

cap = cv2.VideoCapture(0)

prev_time = 0
curr_time = 0
mpFace = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFace.FaceDetection(0.75)

hand = foo.HandDetector()

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # res is for face detection whereas res2 is for hand detection
    res = faceDetection.process(imgRGB)
    res2 = hand.find_hands(img)
    if res.detections:
        for id, detection in enumerate(res.detections):
            #print(id, detection)
            #print(detection.score)
            #print(detection.location_data.relative_bounding_box)
            bbox = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bbox = (int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.height * h), int(bbox.width * w))
            cv2.rectangle(img, bbox, (0, 255, 0), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)} %', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX,\
                        1, (0, 255, 0), 2)
            print(bbox) #-> Bounding box informations
            #mpDraw.draw_detection(img, detection)# -> Classing mediapipe drawing method

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img, f'fps : {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('image', img)
    cv2.waitKey(1)