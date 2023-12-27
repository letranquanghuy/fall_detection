# Predict by python API
from ultralytics import YOLO

model = YOLO("D:/HCMUT/Ths/Thesis/deep_sort/best24_12.engine")
result = model.predict(source='/content/drive/MyDrive/Thesis_Master/YOLO/Yolo_V8/h2(87).jpg')