# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import time
import pandas as pd
import glob

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

def convert_data(keypoints: np.ndarray):
    data = keypoints[:, :-1].flatten().tolist()
    return tuple(data)



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

# type_path = ['east', 'west', 'south', 'north', 'north_east', 'north_west', 'south_east', 'south_west']
type_path = ['east']
model = hub.load('D:\HCMUT\Ths\Thesis\Movenet\movenet_singlepose_thunder_4.tar\movenet_singlepose_thunder_4')
movenet = model.signatures['serving_default']

for type_ in type_path:
    print(type_)
    path = f'D:/HCMUT/Ths/Thesis/Evaluate/{type_}/'
    for label in ["NOT_FALL", "FALL", "LIE"]:
        data_list = []
        n_time_steps = 8
        check = {}

        for video_source in glob.glob(f'D:/HCMUT/Ths/Thesis/Evaluate/{type_}/1/{label}.mp4'):
            print(video_source)
            # Download the model from TF Hub.
            # Threshold for 
            threshold = 0.05

            # Loads video source (0 is for main webcam)
            cap = cv2.VideoCapture(video_source)

            # Checks errors while opening the Video Capture
            if not cap.isOpened():  
                print('Error loading video')
                quit()
            i = 0 
            while True:
                success, img = cap.read()
                last_time = time.time()
                if not success:
                    check[video_source] = i
                    print('Total frame of this video:', i)
                    print('Error reding frame')
                    break
                keypoints = pose_detect(movenet, img, threshold)
                if not isinstance(keypoints, np.ndarray): 
                    continue
                keypoints_data = convert_data(keypoints)
                data_list.append(keypoints_data)
                i += 1
                img = draw_keypoints(img, keypoints)

                fps = 1/(time.time()-last_time)
                img = cv2.putText(img, 'fps: '+ "%.2f"%(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                # Shows image
                cv2.imshow('Movenet', img)
                # Waits for the next frame, checks if q was pressed to quit
                if cv2.waitKey(1) == ord("q"):
                    break
            print("Len data:",len(data_list))  
            len_data = len(data_list)
            if len_data < n_time_steps: 
                continue       
            data_list.append(())
            
            cap.release()
            cv2.destroyAllWindows()

        check_view = [ (v,k) for k,v in check.items() ]
        check_view.sort()
        for v,k in check_view:
            print(f"{k}: {v}")

        # Write vào file csv
        df  = pd.DataFrame(data_list)
        df.to_csv(path + label + ".csv")

