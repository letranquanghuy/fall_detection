from ultralytics import YOLO

# import tensorrt

# print(tensorrt.__version__)

# Load a model
model = YOLO('yolov8/models/train5/weights/best.pt')  # load a custom trained model

# Export the model
model.export(format='engine', half=True, workspace=12)