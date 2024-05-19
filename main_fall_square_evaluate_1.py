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
import glob
RED =   (0, 0, 255) 
GREEN = (0, 255, 0) 
BLUE =  (255, 0, 0) 

# VIDEO SOURCE
# video_path = 0
# video_path = 'D:/HCMUT/Ths/Thesis/deep_sort/data/2_person(2).mp4'


# YOLOv8
# Load model Yolov8
print(torch.cuda.is_available())
weight_path = 'D:/HCMUT/Ths/Thesis/yolov8/tensorrt/best2_1.engine'
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
movenet = modeltf.signatures['serving_default']
keypoints_threshold = 0.05

# LSTM
# Load model LSTM
n_time_steps = 8
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
# with tf.io.gfile.GFile("D:/HCMUT/Ths/Thesis/LSTM/output/frozen_graph_24_12_2023_06_19.pb", "rb") as f:
with tf.io.gfile.GFile("D:/HCMUT/Ths/Thesis/LSTM/output/frozen_graph_07_01_2024_14_36.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(f.read())

# Wrap frozen graph to ConcreteFunctions
frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                inputs=["x:0"],
                                outputs=["Identity:0"],
                                print_graph=True)

tracker = Tracker()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]


def pose_detect(model, img, threshold, x1, y1, x2, y2):
    height, width, _ = img.shape
    if (y2-y1) < (x2-x1):
        y2 += (x2-x1) - (y2-y1)
    elif (y2-y1) > (x2-x1):
        x2 += (y2-y1) - (x2-x1)

    crop_img = img[y1:y2, x1:x2]
    height_crop, width_crop, _ = crop_img.shape
    # A frame of video or an image, represented as an int32 tensor of shape: 256x256x3. Channels order: RGB with values in [0, 255].
    tf_img = cv2.resize(crop_img, (256,256))
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
        keypoints = convert_data(keypoints)
    else:
        keypoints = ()

    return tuple(keypoints)

def draw_keypoints(img, keypoints: np.ndarray, x1, y1, x2, y2):
    if (y2-y1) < (x2-x1):
        y2 += (x2-x1) - (y2-y1)
    elif (y2-y1) > (x2-x1):
        x2 += (y2-y1) - (x2-x1)

    crop_img = img[y1:y2, x1:x2]
    height_crop, width_crop, _ = crop_img.shape
    for i in range(0, len(keypoints),2):
        yc = int(keypoints[i]*height_crop+y1)
        xc = int(keypoints[i+1]*width_crop+x1)

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
    return data

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
        label = "LIE DOWN"
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
        label = "LIE DOWN"
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


def pad_to_square(image):
    height, width = image.shape[:2]
    if height < width:
        diff = width - height
        top = diff // 2
        bottom = diff - top
        left = 0
        right = 0
    else:
        diff = height - width
        top = 0
        bottom = 0
        left = diff // 2
        right = diff - left

    # Define the color for padding (you can change it as needed)
    color = [0, 0, 0]  # Black color padding

    # Apply padding to the image
    squared_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return squared_image

for video_path in glob.glob(f'D:/HCMUT/Ths/Thesis/Evaluate/*/*/*_*_*.mp4'):
    cap = cv2.VideoCapture(video_path)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
    ret, frame = cap.read()

    # Class Name and Colors
    label_map = model.names
    COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in label_map]
    # FPS Detection
    frame_count = 0
    total_fps = 0
    avg_fps = 0
    have_fall = False
    status_people = {}

    while ret:
        start = time.time()
        y, x, _ = frame.shape # height, width, dim

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
            # x1, y1, x2, y2 = list(map(int, bbox))
            x1, y1, x2, y2 = list(map(lambda x: x if x >= 0 else 0,list(map(int, bbox))))
            track_id = track.track_id
            track_id_list.append(track_id)
            if track_id not in status_people.keys():
                status_people[track_id] = {
                                            'data': [], 
                                            'bbox': (x1, y1, x2, y2), 
                                            'status': 'NOT_FALL', 
                                            'is_falled': False, 
                                            'lost_obj_count': 0, 
                                        }
            else:
                if len(status_people[track_id]['data']) == n_time_steps:
                    status_people[track_id]['data'].pop(0)
            # detect pose
            temp_frame = fill_black(pose_frame, x1, y1, x2, y2)
            keypoints = pose_detect(movenet, temp_frame, keypoints_threshold, x1, y1, x2, y2)
            status_people[track_id]['bbox'] = (x1, y1, x2, y2)
            if len(keypoints)!=0:
                frame = draw_keypoints(frame, keypoints, x1, y1, x2, y2)
                status_people[track_id]['data'].append(keypoints)

        # Fall classification
        temp_status_people = copy.deepcopy(status_people)
        def LSTM_predict(track_id):
            if len(status_people[track_id]['data']) == n_time_steps:
                return fall_detect_frozen_graph(frozen_func, status_people[track_id]['data'])
            else:
                return 'None'

        for track_id in temp_status_people.keys():
            if track_id not in track_id_list:
                status_people[track_id]['lost_obj_count'] += 1
            else:
                status_people[track_id]['lost_obj_count'] = 0

            if status_people[track_id]['lost_obj_count'] > 4:
                del status_people[track_id]
                continue
        
        count_not_fall = 0
        for track_id in status_people.keys():
            if len(status_people[track_id]['data']) == n_time_steps:
                # pose_status = fall_detect(model_lstm, status_people[track_id]['data'])
                pose_status = fall_detect_frozen_graph(frozen_func, status_people[track_id]['data'])
                status_people[track_id]['status'] = pose_status

            if track_id in track_id_list:
                if status_people[track_id]['status'] == 'FALL':
                    have_fall = True
                if status_people[track_id]['status'] == 'NOT FALL':
                    count_not_fall += 1
                x1, y1, x2, y2 = status_people[track_id]['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (colors[track_id % len(colors)]), 3)
                frame = cv2.putText(frame, str(track_id), (x1, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (colors[track_id % len(colors)]), 2)
                frame = cv2.putText(frame, status_people[track_id]['status'], (x1+30, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (colors[track_id % len(colors)]), 2)

        if count_not_fall==len(track_id_list):
            have_fall = False

        if have_fall:
            frame = cv2.circle(frame, (x-30, 30), 20, RED, -1) 
            frame = cv2.putText(frame, 'Fall detected!', (x-250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2)
        else:
            frame = cv2.circle(frame, (x-30, 30), 20, GREEN, -1) 
        end = time.time()
        fps = round(1/(end-start))
        frame = cv2.putText(frame, 'fps: '+ "%.2f"%(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()
