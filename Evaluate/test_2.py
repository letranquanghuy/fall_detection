import os
import random
import copy
import cv2
import time

from ultralytics import YOLO
# from deep_sort.tracker import Tracker
import tensorflow as tf
import tensorflow as keras
import tensorflow_hub as hub
import numpy as np
import torch
import modules.utils as utils
from modules.autobackend import AutoBackend
import pathlib
import pandas as pd
import glob

RED =   (0, 0, 255) 
GREEN = (0, 255, 0) 
BLUE =  (255, 0, 0) 


# YOLOv8
# Load model Yolov8
print(torch.cuda.is_available())
weight_path = 'D:/HCMUT/Ths/Thesis/yolov8/tensorrt/best2_1.engine'
file_extension = pathlib.Path(weight_path).suffix
if(file_extension == ".engine"):
    model = AutoBackend(weight_path, device=torch.device('cuda:0'), fp16=True)
    model.warmup()
else:
    model = YOLO(weight_path)
    model.to('cuda')

# define some constants
yolo_threshold = 0.8

# MOVENET
# Load model Movenet
modeltf = hub.load('D:/HCMUT/Ths/Thesis/deep_sort/movenet_singlepose_thunder_4.tar/movenet_singlepose_thunder_4')
movenet = modeltf.signatures['serving_default']
keypoints_threshold = 0.05

# LSTM
# Load model LSTM
status_people = {}
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
# with tf.io.gfile.GFile("D:/HCMUT/Ths/Thesis/LSTM/output/frozen_graph_05_01_2024_10_31.pb", "rb") as f:
with tf.io.gfile.GFile("D:/HCMUT/Ths/Thesis/LSTM/output/frozen_graph_07_01_2024_14_36.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(f.read())

# Wrap frozen graph to ConcreteFunctions
frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                inputs=["x:0"],
                                outputs=["Identity:0"],
                                print_graph=False)

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

type_path = ['1', '2']
# type_path = ['south_west']
for type_ in type_path:
    print(type_)
    for label in ["NOT_FALL", "LIE"]:
        for video_path in glob.glob(f'D:/HCMUT/Ths/Thesis/Evaluate/normal/{type_}/sit_to_chair/*/{label}*.mp4'):
            data_list = []
            path = video_path.split(f'{label}')[0]
            # VIDEO SOURCE
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            # FPS Detection
            total_fps = 0
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
                    x1 = int(x1)
                    x2 = int(x2)
                    y1 = int(y1)
                    y2 = int(y2)

                    temp_frame = fill_black(pose_frame, x1, y1, x2, y2)
                    keypoints = pose_detect(movenet, temp_frame, keypoints_threshold, x1, y1, x2, y2)
                    frame = draw_keypoints(frame, keypoints, x1, y1, x2, y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 3)
                    if len(keypoints)!=0:
                        data_list.append(keypoints)

                end = time.time()
                fps = round(1/(end-start))
                frame = cv2.putText(frame, 'fps: '+ "%.2f"%(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                cv2.imshow('frame', frame)
                cv2.waitKey(1)
                ret, frame = cap.read()
            print(f"{video_path}:",len(data_list))  
            len_data = len(data_list)
            if len_data < n_time_steps:
                continue    


            # Write vào file csv
            df  = pd.DataFrame(data_list)
            df.to_csv(path + label + ".csv")

            cap.release()
            cv2.destroyAllWindows()
            

