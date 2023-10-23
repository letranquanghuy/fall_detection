from ultralytics import YOLO
from PIL import Image
import cv2
import datetime
import torch
print(torch.cuda.is_available())
model = YOLO(r"D:\HCMUT\Ths\Thesis\yolov8\yolov8n.pt")
model.to('cuda')
# results = model.predict(source=r'D:\HCMUT\Ths\Thesis\Movenet\test_1.jpg')
# annotated_frame = results[0].plot()
# # Display the annotated frame
# cv2.imshow("YOLOv8 Inference", annotated_frame)
# cv2.waitKey(0)
# show image with PIL
# for r in results:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#     im.show()  # show image
#     im.save('results.jpg')  # save image

# result = model.predict(show=True, source=0, conf=0.8)

# import numpy as np
# import cv2
# cap = cv2.VideoCapture(0)
# while(True):
#      # Thu lan luot tung khung hinh (frame-by-frame)
#      ret, frame = cap.read()
#      # Thao tac tren frame (chuyen thanh anh xam)
#      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#      # Hien thi frame
#      cv2.imshow('frame', gray)
#      if cv2.waitKey(1) & 0xFF == ord('q'):
#           break
# # Giai phong video khi thuc hien xong thao tac
# cap.release()
# cv2.destroyAllWindows()



cap = cv2.VideoCapture(0)
# define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0) 
while True:
    # start time to compute the fps
    start = datetime.datetime.now()

    ret, frame = cap.read()

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

        # if the confidence is greater than the minimum confidence,
        # draw the bounding box on the frame
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), GREEN, 2)
        # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    total = (end - start).total_seconds()
    print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")

    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / total:.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()