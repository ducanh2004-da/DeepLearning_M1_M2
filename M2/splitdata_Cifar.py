# -*- coding: utf-8 -*-
import os
import shutil
import torchvision
from PIL import Image

final_base_dir = r'D:\work\MachineLearning\DeepLearning\Cifar10\SV2026\SV2026\M2\data_Cifar10'
train_dir = os.path.join(final_base_dir, 'train')
test_dir = os.path.join(final_base_dir, 'test')

if os.path.exists(final_base_dir):
    shutil.rmtree(final_base_dir)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

print("Đang tải dataset CIFAR-10 gốc từ máy chủ...")
trainset = torchvision.datasets.CIFAR10(root='./cifar10_raw', train=True, download=True)
testset = torchvision.datasets.CIFAR10(root='./cifar10_raw', train=False, download=True)

classes = trainset.classes
print(f"Danh sách các classes: {classes}")

def extract_to_folders(dataset, target_dir, split_name):
    print(f"\nĐang giải nén và lưu ảnh vào tập {split_name}...")
    
    for class_name in classes:
        os.makedirs(os.path.join(target_dir, class_name), exist_ok=True)
        
    for i in range(len(dataset)):
        img, label_idx = dataset[i]
        class_name = classes[label_idx]
        
        img_name = f"{class_name}_{i:05d}.png"
        img_path = os.path.join(target_dir, class_name, img_name)
        img.save(img_path)
        
        if (i + 1) % 10000 == 0:
            print(f" - Đã xuất thành công {i + 1}/{len(dataset)} ảnh...")

extract_to_folders(trainset, train_dir, "Train")
extract_to_folders(testset, test_dir, "Test")

if os.path.exists('./cifar10_raw'):
    shutil.rmtree('./cifar10_raw')


print(f"\n✅ Tạo dữ liệu CIFAR-10 thành công!")
print(f"Thư mục lưu trữ: {final_base_dir}")
print(f"Số lượng: {len(trainset)} train, {len(testset)} test samples.")