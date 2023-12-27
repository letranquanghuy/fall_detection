import os
import random
import copy
import cv2
import time

from ultralytics import YOLO
from deep_sort.tracker import Tracker
import tensorflow as tf
import tensorflow as keras
import tensorflow_hub as hub
import numpy as np
import torch
import deep_sort.modules.utils as utils
from deep_sort.modules.autobackend import AutoBackend
import pathlib
import concurrent.futures

RED =   (0, 0, 255) 
GREEN = (0, 255, 0) 
BLUE =  (255, 0, 0) 

# VIDEO SOURCE
video_path = 'D:/HCMUT/Ths/Thesis/deep_sort/data/test_persons_5.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

# YOLOv8
# Load model Yolov8
print(torch.cuda.is_available())
weight_path = 'D:/HCMUT/Ths/Thesis/deep_sort/best12_17.engine'
# weight_path = 'D:/HCMUT/Ths/Thesis/deep_sort/best.pt'
file_extension = pathlib.Path(weight_path).suffix
if(file_extension == ".engine"):
    model = AutoBackend(weight_path, device=torch.device('cuda:0'), fp16=True)
    model.warmup()
else:
    model = YOLO(weight_path)
    model.to('cuda')

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
# model_lstm = tf.keras.models.load_model("D:/HCMUT/Ths/Thesis/LSTM/output/model.h5")

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    if print_graph == True:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        for layer in layers:
            print(layer)
        print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

# Load frozen graph using TensorFlow 1.x functions
with tf.io.gfile.GFile("D:/HCMUT/Ths/Thesis/LSTM/output/frozen_graph_17_12_2023_08_23.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(f.read())

# Wrap frozen graph to ConcreteFunctions
frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                inputs=["x:0"],
                                outputs=["Identity:0"],
                                print_graph=True)

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
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)
    predicted_class = np.argmax(results)

    if predicted_class == 0:     
        label = "NOT FALL"
    elif predicted_class == 1:
        label = "FALL"
    elif predicted_class == 2:
        label = "LIE"
    return label

def fall_detect_frozen_graph(frozen_func, lm_list):
    lm_list = np.array(lm_list)
    input_predict_expand = lm_list[np.newaxis,:]
    input_predict_expand = np.array(input_predict_expand,np.float32)
    frozen_graph_predictions = frozen_func(x=tf.constant(input_predict_expand))[0]
    predicted_class = np.argmax(frozen_graph_predictions)
    if predicted_class == 0:     
        label = "NOT FALL"
    elif predicted_class == 1:
        label = "FALL"
    elif predicted_class == 2:
        label = "LIE"
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
    start = time.time()
    y, x, _ = frame.shape
    pose_frame = copy.deepcopy(frame)
    if(file_extension == ".engine"):
        results = yolo_detect_tensorrt(model, video_path, frame)
    else:
        results = yolo_detect(model, frame)
    # detect person
    detections = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, _ = r
        if float(score) < yolo_threshold:
            continue

        detections.append([int(x1), int(y1), int(x2), int(y2), score])

    # deep sort tracking person
    track_id_list = []
    tracker.update(frame, detections)
    for track in tracker.tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = list(map(int, bbox))
        track_id = track.track_id
        track_id_list.append(track_id)
        if track_id not in status_people.keys():
            status_people[track_id] = {
                                        'data': [], 
                                        'bbox': (x1, y1, x2, y2), 
                                        'status': 'NOT_FALL', 
                                        'is_falled': False, 
                                        'lost_obj_count': 0, 
                                        'recover_count': 0,
                                       }
        else:
            if len(status_people[track_id]['data']) == n_time_steps:
                status_people[track_id]['data'].pop(0)
        # detect pose
        temp_frame = fill_black(pose_frame, x1, y1, x2, y2)
        keypoints = pose_detect(movenet, temp_frame, keypoints_threshold)
        # frame = draw_keypoints(frame, keypoints)
        if isinstance(keypoints, np.ndarray): 
            keypoints = convert_data(keypoints)
            status_people[track_id]['data'].append(keypoints)
            status_people[track_id]['bbox'] = (x1, y1, x2, y2)

    # Fall classification
    temp_status_people = copy.deepcopy(status_people)
    for track_id in temp_status_people.keys():
        x1, y1, x2, y2 = status_people[track_id]['bbox']
        if track_id not in track_id_list:
            status_people[track_id]['lost_obj_count'] += 1
        else:
            status_people[track_id]['lost_obj_count'] = 0

        if status_people[track_id]['lost_obj_count'] > 4:
            del status_people[track_id]
            continue

        if len(status_people[track_id]['data']) == n_time_steps:
            # pose_status = fall_detect(model_lstm, status_people[track_id]['data'])
            pose_status = fall_detect_frozen_graph(frozen_func, status_people[track_id]['data'])
            status_people[track_id]['status'] = pose_status

            # To avoid noise affecting the pose detection result, we will use the recover counter
            # After fall if status change to not fall, recover counter will count
            # If recover counter equal 5 consecutive frame, status will change to NOT FALL
            # if status_people[track_id]['is_falled'] and pose_status == 'NOT FALL':
            #     status_people[track_id]['recover_count'] += 1

            # if pose_status == 'FALL':
            #     status_people[track_id]['is_falled'] = True
            #     status_people[track_id]['recover_count'] = 0
            # elif status_people[track_id]['recover_count'] == 5 and pose_status == 'NOT FALL':
            #     status_people[track_id]['is_falled'] = False

            # if status_people[track_id]['is_falled']:
            #     status_people[track_id]['status'] = 'FALL'
            # else:
            #     status_people[track_id]['status'] = pose_status
        
        if track_id in track_id_list:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (colors[track_id % len(colors)]), 3)
            frame = cv2.putText(frame, str(track_id), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (colors[track_id % len(colors)]), 2)
            frame = cv2.putText(frame, status_people[track_id]['status'], (x1+30, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (colors[track_id % len(colors)]), 2)


    end = time.time()
    fps = round(1/(end-start))
    frame = cv2.putText(frame, 'fps: '+ "%.2f"%(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
