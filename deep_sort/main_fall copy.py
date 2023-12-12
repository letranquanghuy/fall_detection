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
import modules.utils as utils
from modules.autobackend import AutoBackend
import pathlib


RED =   (0, 0, 255) 
GREEN = (0, 255, 0) 
BLUE =  (255, 0, 0) 

# VIDEO SOURCE
video_path = 'D:/HCMUT/Ths/Thesis/deep_sort/data/test.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

# YOLOv8
# Load model Yolov8
print(torch.cuda.is_available())
weight_path = 'D:/HCMUT/Ths/Thesis/deep_sort/best.engine'
# weight_path = 'D:/HCMUT/Ths/Thesis/deep_sort/best.pt'
file_extension = pathlib.Path(weight_path).suffix
if(file_extension == ".engine"):
    model = AutoBackend(weight_path, device=torch.device('cuda:0'), fp16=True)
    model.warmup()
else:
    model = YOLO(weight_path)
    model.to('cuda')
# define some constants
CONFIDENCE_THRESHOLD = 0.8
# Warmup

# define some constants
yolo_threshold = 0.7

# MOVENET
# Load model Movenet
modeltf = hub.load('D:/HCMUT/Ths/Thesis/deep_sort/movenet_singlepose_thunder_4.tar/movenet_singlepose_thunder_4')
# modeltf = hub.load('D:/HCMUT/Ths/Thesis/deep_sort/movenet_singlepose_lightning_4.tar/movenet_singlepose_lightning_4')
movenet = modeltf.signatures['serving_default']
keypoints_threshold = 0.05

# LSTM
# Load model LSTM
status_people = {}
n_time_steps = 10
model_lstm = tf.keras.models.load_model("D:/HCMUT/Ths/Thesis/LSTM/output/model.h5")

tracker = Tracker()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

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

def fall_detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)
    predicted_class = np.argmax(results)

    if predicted_class == 1:     
        label = "FALL"
    else:
        label = "NOT FALL"
    return label


def yolo_detect_tensorrt(model, source, image):
    # Preprocess
    im = utils.preprocess(image)

    # Inference
    preds = model(im)

    # Post Process
    results = utils.postprocess(preds, im, image, model.names, source)
    return results[0]

def yolo_detect(model, image):
    return model(image)[0]

# Class Name and Colors
label_map = model.names
COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in label_map]
# FPS Detection
frame_count = 0
total_fps = 0
avg_fps = 0
while ret:
    last_time = time.time()
    y, x, _ = frame.shape
    ret, frame = cap.read()
    pose_frame = copy.deepcopy(frame)
    if(file_extension == ".engine"):
        detections = yolo_detect_tensorrt(model, video_path, frame)
    else:
        detections = yolo_detect(model, frame)

    # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the detection
        confidence = data[4]

        # filter out weak detections by ensuring the 
        # confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])

        # temp_frame = fill_black(pose_frame, xmin, ymin, xmax, ymax)
        # cv2.imshow("test", temp_frame)
        # keypoints = pose_detect(movenet, temp_frame, threshold)

        # frame = draw_keypoints(frame, keypoints)

        # if the confidence is greater than the minimum confidence,
        # draw the bounding box on the frame
        cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), GREEN, 2)

    # calculate the frame per second and draw it on the frame
    fps = 1/(time.time()-last_time)
    frame = cv2.putText(frame, 'fps: '+ "%.2f"%(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
