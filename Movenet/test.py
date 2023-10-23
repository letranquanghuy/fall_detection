import cv2
video_source = 0
cap = cv2.VideoCapture(video_source)

success, img = cap.read()


while True:
    cv2.imshow("test", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break