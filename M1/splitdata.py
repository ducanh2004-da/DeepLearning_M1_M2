# -*- coding: utf-8 -*-

import os
import shutil
import kagglehub
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Download dataset (hoặc load từ cache nếu đã tải)
path = kagglehub.dataset_download("shiv28/animal-5-mammal")
print("Path to dataset files:", path)

# 2. Cập nhật thư mục gốc chứa ảnh
search_root = os.path.join(path, 'Animal')
if not os.path.exists(search_root):
    search_root = path  # Fallback nếu cấu trúc thư mục thay đổi

print(f"Searching for images in: {search_root}")

# 3. Đường dẫn đích bạn yêu cầu để lưu folder data
final_base_dir = r'D:\work\MachineLearning\DeepLearning\Cifar10\SV2026\SV2026\M1\data'
train_dir = os.path.join(final_base_dir, 'train')
test_dir = os.path.join(final_base_dir, 'test')

# Xóa folder data cũ nếu đã tồn tại để tránh bị lẫn lộn dữ liệu khi chạy lại code nhiều lần
if os.path.exists(final_base_dir):
    shutil.rmtree(final_base_dir)

all_paths = []
all_labels = []

# Quét tất cả thư mục con
sub_dirs = [d for d in os.listdir(search_root) if os.path.isdir(os.path.join(search_root, d))]

for sub in sub_dirs:
    sub_path = os.path.join(search_root, sub)
    images = [f for f in os.listdir(sub_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Nếu thư mục hiện tại chứa trực tiếp hình ảnh
    if len(images) > 0:
        for img in images:
            all_paths.append(os.path.join(sub_path, img))
            all_labels.append(sub)
    else:
        # Nếu thư mục chứa các subfolder class (VD: train/cat, val/dog)
        for cat in os.listdir(sub_path):
            cat_path = os.path.join(sub_path, cat)
            if os.path.isdir(cat_path):
                images = [f for f in os.listdir(cat_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if len(images) > 0:
                    for img in images:
                        all_paths.append(os.path.join(cat_path, img))
                        all_labels.append(cat)

# 4. Thực hiện split và copy dữ liệu
if len(all_paths) == 0:
    print(f"Error: No images found. Directory content: {os.listdir(search_root)}")
else:
    # Chia 80% train, 20% test và phân bổ đều các class (stratify)
    tr_p, te_p, tr_l, te_l = train_test_split(
        all_paths, 
        all_labels, 
        test_size=0.2, 
        stratify=all_labels, 
        random_state=42
    )

    print("\nĐang tạo và copy dữ liệu vào tập Train...")
    for p, l in zip(tr_p, tr_l):
        os.makedirs(os.path.join(train_dir, l), exist_ok=True)
        shutil.copy(p, os.path.join(train_dir, l, os.path.basename(p)))
        
    print("Đang tạo và copy dữ liệu vào tập Test...")
    for p, l in zip(te_p, te_l):
        os.makedirs(os.path.join(test_dir, l), exist_ok=True)
        shutil.copy(p, os.path.join(test_dir, l, os.path.basename(p)))

    # Tổng kết
    print(f"\n✅ Split Data thành công!")
    print(f"Thư mục lưu trữ: {final_base_dir}")
    print(f"Số lượng: {len(tr_p)} train, {len(te_p)} test samples.")
    print(f"Các Classes đã tìm thấy: {np.unique(all_labels)}")