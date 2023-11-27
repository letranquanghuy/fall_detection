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

video_path = os.path.join('.', 'data', 'test.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

print(torch.cuda.is_available())
model = YOLO("best.pt")
model.to('cuda')

# define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0) 

modeltf = hub.load('D:\HCMUT\Ths\object-tracking-yolov8-deep-sort\movenet_singlepose_thunder_4.tar\movenet_singlepose_thunder_4')
movenet = modeltf.signatures['serving_default']
# Threshold for 
threshold = 0.05

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


tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5
while ret:
    start = time.time()
    results = model(frame)[0]
    y, x, _ = frame.shape
    pose_frame = copy.deepcopy(frame)

    detections = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, _ = r
        if float(score) < CONFIDENCE_THRESHOLD:
            continue
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)

        if score > detection_threshold:
            detections.append([x1, y1, x2, y2, score])
        temp_frame = fill_black(pose_frame, x1, y1, x2, y2)
        keypoints = pose_detect(movenet, temp_frame, threshold)

        frame = draw_keypoints(frame, keypoints)
    tracker.update(frame, detections)

    for track in tracker.tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = bbox
        track_id = track.track_id
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
    end = time.time()
    fps = round(1/(end-start))
    frame = cv2.putText(frame, 'fps: '+ "%.2f"%(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
