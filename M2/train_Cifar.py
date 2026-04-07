# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv
import os
from datetime import datetime
from models.model_M2 import M1_CNN

# === CONFIGURATION ===
IMG_SIZE = 32 # Tối ưu cho CIFAR-10 (ảnh gốc 32x32)
BATCH_SIZE = 32
EPOCHS = 200
# Trỏ tới thư mục CIFAR-10
DATA_DIR = r'D:\work\MachineLearning\DeepLearning\Cifar10\SV2026\SV2026\M2\data_Cifar10'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESUME_TRAINING = True # Đổi thành True nếu bạn muốn train tiếp từ checkpoint cũ

print(f"Training on device: {DEVICE}")

# === DATALOADER ===
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.ImageFolder(f'{DATA_DIR}/train', transform=train_transform)
test_data = datasets.ImageFolder(f'{DATA_DIR}/test', transform=eval_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# === MODEL SETUP ===
# Cực kỳ quan trọng: CIFAR-10 có 10 classes
model = M1_CNN(variant=IMG_SIZE, num_classes=10).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# === CHECKPOINT & RESUME LOGIC ===
# Đổi tên file lưu trữ để không bị lẫn với project cũ
checkpoint_path = f'best_M1_CIFAR10_{IMG_SIZE}.pth'
log_file = f'training_log_CIFAR10_{IMG_SIZE}.csv'

best_test_acc = 0.0
best_epoch = 0
start_epoch = 0

# Nếu bật RESUME_TRAINING và file checkpoint tồn tại, tiến hành load lại trạng thái
if RESUME_TRAINING and os.path.exists(checkpoint_path):
    print(f"[*] Đang tải lại checkpoint từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_test_acc = checkpoint['best_acc']
    best_epoch = checkpoint.get('best_epoch', start_epoch)
    print(f"[*] Đã tải thành công. Tiếp tục huấn luyện từ Epoch {start_epoch+1}. Best Acc hiện tại: {best_test_acc:.2f}%\n")
else:
    # Nếu train từ đầu, khởi tạo lại file log
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc'])

# === TRAINING LOOP ===
# Mở file log ở chế độ 'a' (append) để ghi tiếp nếu đang resume
with open(log_file, mode='a', newline='') as file:
    writer = csv.writer(file)

    for epoch in range(start_epoch, EPOCHS):
        # ---------- TRAIN PHASE ----------
        model.train()
        train_loss, train_correct, total_train = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * train_correct / total_train
        train_loss = train_loss / len(train_loader)

        # ---------- EVALUATION (TEST) PHASE ----------
        model.eval()
        test_loss, test_correct, total_test = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
        test_acc_epoch = 100. * test_correct / total_test
        test_loss = test_loss / len(test_loader)
        scheduler.step()

        # --- LƯU CHECKPOINT ĐỂ RESUME SAU NÀY ---
        if test_acc_epoch > best_test_acc:
            best_test_acc = test_acc_epoch
            best_epoch = epoch + 1
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_test_acc,
                'best_epoch': best_epoch
            }
            torch.save(checkpoint, checkpoint_path)

        # --- LOGGING FORMAT THEO YÊU CẦU ---
        current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f"{current_time} Epoch {epoch+1}/{EPOCHS} summary:  loss_train={train_loss:.5f},  acc_train={train_acc:.2f}%,  loss_test={test_loss:.5f},  acc_test={test_acc_epoch:.2f}% (best: {best_test_acc:.2f}% @ epoch {best_epoch})")
        
        # Ghi vào file csv
        writer.writerow([epoch+1, train_loss, train_acc, test_loss, test_acc_epoch])
            
print(f"\nQuá trình huấn luyện hoàn tất! Best Test Accuracy: {best_test_acc:.2f}% tại epoch {best_epoch}")

# === FINAL TESTING PHASE ===
print("\n--- Đang load lại mô hình tốt nhất để Final Test ---")
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

final_test_loss, final_test_correct, final_total_test = 0, 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        final_test_loss += loss.item()
        _, predicted = outputs.max(1)
        final_total_test += labels.size(0)
        final_test_correct += predicted.eq(labels).sum().item()

final_test_acc = 100. * final_test_correct / final_total_test
final_test_loss = final_test_loss / len(test_loader)

print(f"Final Test Loss: {final_test_loss:.4f} | Final Test Accuracy: {final_test_acc:.2f}%")