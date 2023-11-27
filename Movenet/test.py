import cv2
video_source = 'D:/HCMUT/Ths/Thesis/Movenet/testLSTM2.mp4'
cap = cv2.VideoCapture(video_source)

success, img = cap.read()


while True:
    success, img = cap.read()

    cv2.imshow("test", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break