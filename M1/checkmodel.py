# -*- coding: utf-8 -*-
import torch
from torchsummary import summary

# Import model của bạn (Đảm bảo đường dẫn import khớp với cấu trúc thư mục của bạn)
from models.model_M1 import M1_CNN 

# === CẤU HÌNH ===
# Đang set mặc định cho CIFAR-10. 
# Nếu muốn check lại bản cũ, bạn đổi thành IMG_SIZE = 224 và NUM_CLASSES = 5
IMG_SIZE = 32
NUM_CLASSES = 10

print(f"=== KIỂM TRA MÔ HÌNH M1_CNN (Variant: {IMG_SIZE}x{IMG_SIZE}) ===")

# 1. Khởi tạo mô hình
model = M1_CNN(variant=IMG_SIZE, num_classes=NUM_CLASSES)

# 2. Chuyển mô hình lên GPU (nếu có) để torchsummary hoạt động chính xác
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 3. In cấu trúc cơ bản của mô hình bằng hàm print mặc định
print("\n[1] CẤU TRÚC MÔ HÌNH (DẠNG RAW):")
print(model)

# 4. Tính toán số lượng tham số (Parameters)
# Mình thêm phần tách biệt giữa tham số có thể train và không thể train (như BatchNorm)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n[2] THÔNG TIN THAM SỐ:")
print(f" - Tổng số tham số (Total Params): {total_params:,}")
print(f" - Số tham số huấn luyện được (Trainable Params): {trainable_params:,}")

# 5. Sử dụng torchsummary để xem chi tiết Output Shape qua từng Layer
print("\n[3] BẢNG TÓM TẮT CHI TIẾT TỪ TORCHSUMMARY:")
try:
    summary(model, (3, IMG_SIZE, IMG_SIZE))
except Exception as e:
    print(f"\n❌ Lỗi khi chạy torchsummary: {e}")
    print("💡 Mẹo: Lỗi ở bước này thường do kích thước Tensor đi qua các layer (đặc biệt là MaxPool hoặc BatchNorm) bị lệch so với thiết kế của BlockBT3. Hãy check lại file BT3.py nếu gặp lỗi nha!")