import torch

# Đường dẫn tới file checkpoint của bạn (nhớ sửa lại nếu bạn lưu ở thư mục khác)
checkpoint_path = r'D:\work\MachineLearning\DeepLearning\Cifar10\SV2026\SV2026\M2\checkpoint(pth)\best_M2_CIFAR10_32.pth'

# Đọc file bằng torch.load
# Lưu ý: Thêm map_location='cpu' là một thói quen tốt để có thể mở file trên bất kỳ máy nào, kể cả máy không có GPU
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

print("=== KIỂM TRA NỘI DUNG CHECKPOINT ===")

# 1. Xem file này đang lưu những cục dữ liệu nào (Keys)
print("\n[1] Các thành phần được lưu:")
print(checkpoint.keys())

# 2. In ra các thông số huấn luyện (Metadata) mà chúng ta đã setup lúc train
print("\n[2] Thông tin lịch sử huấn luyện:")
print(f" - Epoch đang chạy dở: {checkpoint.get('epoch', 'Không có')}")
print(f" - Epoch tốt nhất (Best Epoch): {checkpoint.get('best_epoch', 'Không có')}")
print(f" - Độ chính xác cao nhất (Best Acc): {checkpoint.get('best_acc', 0):.2f}%")

# 3. Xem danh sách các layer và kích thước ma trận trọng số (Weights)
print("\n[3] Danh sách 5 Layer đầu tiên trong model_state_dict:")
# Chỉ in 5 layer đầu để tránh dài dòng (bạn có thể bỏ [list(...)[0:5]] để in toàn bộ)
for param_tensor in list(checkpoint['model_state_dict'].keys())[0:5]:
    size = checkpoint['model_state_dict'][param_tensor].size()
    print(f" - Layer: {param_tensor: <30} | Kích thước: {list(size)}")