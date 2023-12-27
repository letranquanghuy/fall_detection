import cv2
import time
import glob

def video_to_frames(video_path):
    # extract frames from a video and save to directory as 'x.png' where 
    # x is the frame index
    cap = cv2.VideoCapture(video_path)
    count = 0
    global i
    ret, frame = cap.read()
    while ret:
        if count%5 == 0:
            output_path = f'D:/HCMUT/Ths/video/image/1224_{i}.png'
            # output_path = f'D:/HCMUT/Ths/video/image/{int(time.time())}.png'
            i += 1
            cv2.imwrite(output_path, frame)
            count = 0
        count += 1
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()

i = 0 
for video_source in glob.glob(f'D:/HCMUT/Ths/video/*.mp4'):
    video_to_frames(video_source)