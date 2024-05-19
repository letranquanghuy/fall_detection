from ultralytics import YOLO

# import tensorrt

# print(tensorrt.__version__)

# Load a model
model = YOLO('D:/HCMUT/Ths/Thesis/yolov8/tensorrt/best17_1.pt')  # load a custom trained model

# Export the model
model.export(format='engine', half=True, workspace=4)