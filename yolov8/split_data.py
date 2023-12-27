import glob
import random
import shutil

# Đường dẫn tới các ảnh trong thư mục train
train_image_paths = glob.glob(r'D:\HCMUT\Ths\video\image\ok\*.png') + glob.glob(r'D:\HCMUT\Ths\video\image\ok\*.jpg')

# Tính số lượng ảnh cần di chuyển sang test (12.5%)
num_images_to_move = int(len(train_image_paths) * 0.125)

# Lấy ngẫu nhiên các đường dẫn ảnh
random.seed(42)  # Để việc lấy ngẫu nhiên luôn sinh ra kết quả giống nhau
test_image_paths = random.sample(train_image_paths, 148)

print(len(test_image_paths))
# Di chuyển các ảnh sang thư mục test
for image_path in test_image_paths:
    image_name = image_path.split('\\')[-1]  # Tên của ảnh
    print(image_name)
    label_name = image_name.split('.')[0] + '.txt'
    print(label_name)
    destination_path = fr'D:\HCMUT\Ths\Thesis\yolov8\data\test\images\{image_name}'
    shutil.move(image_path, destination_path)
    shutil.move(fr'D:\HCMUT\Ths\video\label\{label_name}', fr'D:\HCMUT\Ths\Thesis\yolov8\data\test\labels\{label_name}')