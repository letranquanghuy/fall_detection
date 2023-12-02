import os
import random
import copy
import cv2
import time

from ultralytics import YOLO
from tracker import Tracker
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import torch

video_path = 'D:/HCMUT/Ths/Thesis/deep_sort/data/test.mp4'

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

print(torch.cuda.is_available())
model = YOLO("D:/HCMUT/Ths/Thesis/deep_sort/best.pt")
model.to('cuda')

# define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0) 

modeltf = hub.load('D:/HCMUT/Ths/object-tracking-yolov8-deep-sort/movenet_singlepose_thunder_4.tar/movenet_singlepose_thunder_4')
movenet = modeltf.signatures['serving_default']
# Threshold for 
threshold = 0.05

tracker = Tracker()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

status_people = {}
n_time_steps = 10

def pose_detect(model, img, threshold):
    # A frame of video or an image, represented as an int32 tensor of shape: 256x256x3. Channels order: RGB with values in [0, 255].
    tf_img = cv2.resize(img, (256,256))
    tf_img = cv2.cvtColor(tf_img, cv2.COLOR_BGR2RGB)
    tf_img = np.asarray(tf_img)
    tf_img = np.expand_dims(tf_img,axis=0)

    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf_img, dtype=tf.int32)

    # Run model inference.
    outputs = model(image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints = outputs['output_0']
    keypoints = keypoints.numpy()[0,0]
    if all(k[2] >= threshold for k in keypoints):
        return keypoints
    else:
        return None

def draw_keypoints(img, keypoints: np.ndarray):
    y, x, _ = img.shape

    # iterate through keypoints
    if isinstance(keypoints, np.ndarray):
        for i in range(len(keypoints)):
            k = keypoints[i]
            # The first two channels of the last dimension represents the yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) of the 17 keypoints
            yc = int(k[0] * y)
            xc = int(k[1] * x)

            # Draws a circle on the image for each keypoint
            img = cv2.circle(img, (xc, yc), 2, (0, 255, 0), 5)
    
    return img

def fill_black(img, xmin, ymin, xmax, ymax):
    img_process = copy.deepcopy(img)
    temp = cv2.rectangle(img_process, (xmin, ymin) , (xmax, ymax), (0, 0, 0), -1)
    output = img-temp
    return output

def convert_data(keypoints: np.ndarray):
    data = keypoints[:, :-1].flatten().tolist()
    return tuple(data)

while ret:
    start = time.time()
    results = model(frame)[0]
    y, x, _ = frame.shape
    pose_frame = copy.deepcopy(frame)
    
    # detect person
    detections = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, _ = r
        if float(score) < CONFIDENCE_THRESHOLD:
            continue
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)

        detections.append([x1, y1, x2, y2, score])

    track_id_list = []
    # deep sort tracking person
    tracker.update(frame, detections)
    for track in tracker.tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = bbox
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        track_id = track.track_id
        track_id_list.append(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (colors[track_id % len(colors)]), 3)

        if track_id not in status_people.keys():
            status_people[track_id] = {"data": [], "count": 0}
        else:
            if len(status_people[track_id]['data']) == n_time_steps:
                status_people[track_id]['data'].pop(0)
        # detect pose
        temp_frame = fill_black(pose_frame, x1, y1, x2, y2)
        keypoints = pose_detect(movenet, temp_frame, threshold)
        frame = draw_keypoints(frame, keypoints)
        if isinstance(keypoints, np.ndarray): 
            keypoints = convert_data(keypoints)
            status_people[track_id]['data'].append(keypoints)

    print(track_id_list)
    temp_status_people = copy.deepcopy(status_people)
    for track_id_ in temp_status_people:
        if track_id_ not in track_id_list:
            status_people[track_id_]['count'] += 1
        else:
            status_people[track_id_]['count'] = 0

        if status_people[track_id_]['count'] > 4:
            del status_people[track_id_]
    print(status_people)
    end = time.time()
    fps = round(1/(end-start))
    frame = cv2.putText(frame, 'fps: '+ "%.2f"%(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
