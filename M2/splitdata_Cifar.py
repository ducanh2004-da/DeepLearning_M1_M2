# -*- coding: utf-8 -*-
import os
import shutil
import torchvision
from PIL import Image

# 1. Cấu hình thư mục đích cho CIFAR-10
final_base_dir = r'D:\work\MachineLearning\DeepLearning\Cifar10\SV2026\SV2026\M2\data_Cifar10'
train_dir = os.path.join(final_base_dir, 'train')
test_dir = os.path.join(final_base_dir, 'test')

# Xóa folder cũ nếu có để tránh bị lẫn dữ liệu
if os.path.exists(final_base_dir):
    shutil.rmtree(final_base_dir)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 2. Tải dataset CIFAR-10 thông qua torchvision
print("Đang tải dataset CIFAR-10 gốc từ máy chủ...")
# Tải tạm vào thư mục con cifar10_raw ở cùng thư mục chạy lệnh
trainset = torchvision.datasets.CIFAR10(root='./cifar10_raw', train=True, download=True)
testset = torchvision.datasets.CIFAR10(root='./cifar10_raw', train=False, download=True)

classes = trainset.classes
print(f"Danh sách các classes: {classes}")

# 3. Hàm hỗ trợ trích xuất ảnh từ Dataset ra thư mục ImageFolder
def extract_to_folders(dataset, target_dir, split_name):
    print(f"\nĐang giải nén và lưu ảnh vào tập {split_name}...")
    
    # Tạo sẵn các thư mục con cho từng class
    for class_name in classes:
        os.makedirs(os.path.join(target_dir, class_name), exist_ok=True)
        
    # Duyệt qua từng ảnh và lưu thành file .png
    for i in range(len(dataset)):
        img, label_idx = dataset[i]
        class_name = classes[label_idx]
        
        # Lưu ảnh với format tên để dễ quản lý
        img_name = f"{class_name}_{i:05d}.png"
        img_path = os.path.join(target_dir, class_name, img_name)
        img.save(img_path)
        
        if (i + 1) % 10000 == 0:
            print(f" - Đã xuất thành công {i + 1}/{len(dataset)} ảnh...")

# 4. Thực thi trích xuất
extract_to_folders(trainset, train_dir, "Train")
extract_to_folders(testset, test_dir, "Test")

# Xóa thư mục raw tải tạm ban đầu cho gọn ổ cứng
if os.path.exists('./cifar10_raw'):
    shutil.rmtree('./cifar10_raw')

# Tổng kết
print(f"\n✅ Tạo dữ liệu CIFAR-10 thành công!")
print(f"Thư mục lưu trữ: {final_base_dir}")
print(f"Số lượng: {len(trainset)} train, {len(testset)} test samples.")