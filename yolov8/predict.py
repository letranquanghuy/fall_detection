from ultralytics import YOLO
from PIL import Image
import cv2
import datetime
import torch
import time
# from Movenet import pose_estimate
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import copy

print(torch.cuda.is_available())
model = YOLO(r"D:\HCMUT\Ths\Thesis\yolov8\models\best.pt")
model.to('cuda')

cap = cv2.VideoCapture(r'D:\HCMUT\Ths\Thesis\yolov8\test.mp4')
# define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0) 

modeltf = hub.load('D:\HCMUT\Ths\Thesis\Movenet\movenet_singlepose_thunder_4.tar\movenet_singlepose_thunder_4')
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

while True:
    # start time to compute the fps
    last_time = time.time()
    ret, frame = cap.read()
    pose_frame = copy.deepcopy(frame)
    # if there are no more frames to process, break out of the loop
    if not ret:
        break

    # run the YOLO model on the frame
    detections = model(frame)[0]

            # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the detection
        confidence = data[4]

        # filter out weak detections by ensuring the 
        # confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])

        temp_frame = fill_black(pose_frame, xmin, ymin, xmax, ymax)
        cv2.imshow("test", temp_frame)
        keypoints = pose_detect(movenet, temp_frame, threshold)

        frame = draw_keypoints(frame, keypoints)
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