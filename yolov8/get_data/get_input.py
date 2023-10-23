import cv2 
import time

vid_capture = cv2.VideoCapture("D:\HCMUT\Ths\Thesis\yolov8\get_data\WIN_20231003_11_59_46_Pro.mp4")
# vid_capture = cv2.VideoCapture("D:\HCMUT\Ths\Thesis\yolov8\get_data\cut_video.mp4")
i = 0
count = 0
while True: 
      # vid_capture.read() methods returns a tuple, first element is a bool 
    # and the second is frame
    ret, frame = vid_capture.read()
    if ret == True:
        cv2.imshow('Frame',frame)
        image_name = str(round(time.time())) + str(i)
        if count%20 == 0:
            cv2.imwrite(f"get_data/output/{image_name}.png", frame)
            print(i, image_name)
            i += 1
        count += 1
        # 20 is in milliseconds, try to increase the value, say 50 and observe
        key = cv2.waitKey(20)
            
        if key == ord('q'):
            break
    else:
        break