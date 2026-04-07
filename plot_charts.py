# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics(csv_file, size):
    # 1. Kiểm tra xem file có tồn tại không để tránh lỗi
    if not os.path.exists(csv_file):
        print(f"Lỗi: Không tìm thấy file '{csv_file}'")
        return

    # 2. SỬA LỖI: Sử dụng tham số csv_file thay vì fix cứng tên file
    df = pd.read_csv(csv_file)
    
    # 3. TỐI ƯU: Tự động lấy số lượng epoch dựa trên số dòng của file csv
    num_epochs = len(df)
    
    plt.figure(figsize=(12, 5))
    
    # Biểu đồ 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(df['Epoch'], df['Train Acc'], label='Train Accuracy', color='blue')
    plt.plot(df['Epoch'], df['Test Acc'], label='Test Accuracy', color='orange')
    plt.title(f'Accuracy over {num_epochs} Epochs (Size {size}x{size})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Biểu đồ 2: Loss
    plt.subplot(1, 2, 2)
    plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss', color='blue')
    plt.plot(df['Epoch'], df['Test Loss'], label='Test Loss', color='orange')
    plt.title(f'Loss over {num_epochs} Epochs (Size {size}x{size})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'chart_{size}.png', dpi=300) # Thêm dpi=300 để ảnh lưu ra nét hơn
    print(f"Đã lưu biểu đồ thành công: chart_{size}.png")
    plt.show()

# Chạy thử nghiệm
plot_metrics('training_log_32.csv', 32)