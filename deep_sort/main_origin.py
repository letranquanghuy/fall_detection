import os
import random
import copy
import cv2
import time

from ultralytics import YOLO
from tracker import Tracker

video_path = os.path.join('.', 'data', 'test.mp4')
video_out_path = os.path.join('.', 'out.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model = YOLO("best.pt")

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5
while ret:
    start = time.time()
    results = model(frame)
    y, x, _ = frame.shape
    frame_out = copy.deepcopy(frame)
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame_out, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,0,0), thickness=-1)
            frame_out = frame-frame_out
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
    end = time.time()
    frame = cv2.putText(img=frame, text=f'{round(1/(end-start))} fps', fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1, org=(0, 20), color=(255, 0, 0))
    
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    cap_out.write(frame_out)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()
