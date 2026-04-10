# -*- coding: utf-8 -*-
import torch
from torchsummary import summary

from models.model_M2 import M1_CNN 

IMG_SIZE = 32
NUM_CLASSES = 10

print(f"=== KIỂM TRA MÔ HÌNH M2_CNN (Variant: {IMG_SIZE}x{IMG_SIZE}) ===")

model = M1_CNN(variant=IMG_SIZE, num_classes=NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("\n[1] CẤU TRÚC MÔ HÌNH (DẠNG RAW):")
print(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n[2] THÔNG TIN THAM SỐ:")
print(f" - Tổng số tham số (Total Params): {total_params:,}")
print(f" - Số tham số huấn luyện được (Trainable Params): {trainable_params:,}")

print("\n[3] BẢNG TÓM TẮT CHI TIẾT TỪ TORCHSUMMARY:")
try:
    summary(model, (3, IMG_SIZE, IMG_SIZE))
except Exception as e:
    print(f"\n❌ Lỗi khi chạy torchsummary: {e}")
    print("💡 Mẹo: Lỗi ở bước này thường do kích thước Tensor đi qua các layer (đặc biệt là MaxPool hoặc BatchNorm) bị lệch so với thiết kế của BlockBT3. Hãy check lại file BT3.py nếu gặp lỗi nha!")