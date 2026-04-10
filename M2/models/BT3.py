# -*- coding: utf-8 -*-
"""
Mã nguồn NetBT3 được chỉnh sửa để khớp với sơ đồ bảng trắng, tối ưu hóa cho cả ảnh kích thước lớn (224x224) và nhỏ (32x32 - CIFAR).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# SE giúp nhận biết kênh nào quan trọng
class SE(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BlockBT3(nn.Module):
    def __init__(self, in_channels, out_channels, s):
        super().__init__()
        self.Pw1 = nn.Conv2d(in_channels = in_channels, out_channels=in_channels, kernel_size=1)
        # Sơ đồ: Dw kernel=3x3, P=1, stride=s
        self.Dw = nn.Conv2d(in_channels = in_channels, out_channels=in_channels, kernel_size=3, stride=s, padding=1, groups=in_channels)
        self.Pw2 = nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size=1)
        # Sơ đồ: Residual PDP + input. Gốc code sử dụng conv based shortcuts
        self.PwR = nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size=1, stride=s)
        
        # Giả định reduction_ratio cho SE là 16
        self.SE = SE(out_channels, reduction_ratio=16) 
        self.s = s

    def forward(self, x):
        identity = x
        
        Pw1 = F.relu(self.Pw1(x))
        Dw = F.relu(self.Dw(Pw1))
        Pw2 = self.Pw2(Dw) # Pw2 output. Gốc code có F.relu(Pw2(Dw)). Tôi loại bỏ F.relu để PDP là output của multiplication.
        
        # att = self.SE(Pw2). Gốc code nhân. Lớp SE standard trả về feature map tỷ lệ.
        # PDP = Pw2 * att
        
        # Tôi sẽ điều chỉnh PDP là output của SE module chuẩn (returns multiplied result)
        PDP = self.SE(Pw2)

        # Sửa logic so sánh shape và stride cho residual connection tối ưu hơn.
        # Gốc code: if self.s == 1 and PDP.size()==x.size(): PDP + x else: PDP + PwR(x).
        # C_out == C_in and s==1 cho identity residual. C_out != C_in or s!=1 cho conv residual.
        if self.s == 1 and x.size(1) == PDP.size(1): # Stride 1 và channels khớp
             x = PDP + identity # Residual identity shortcut
        else:
            # Stride != 1 hoặc channels khác. Conv residual. PwR có stride s.
             x = PDP + self.PwR(x) # Residual conv shortcut
        return F.relu(x)
                
class NetBT3(nn.Module):
    def __init__(self, size_image, n_class):
        super().__init__()
        self.size_image = size_image
        self.n_class = n_class

        # ──────────────── Thiết lập stride tĩnh tối ưu ────────────────
        # Sơ đồ bảng: Stem Conv S=2 (size 112), Blocks 2, 5, 8, 9 S=2. Cuối feature 7x7.
        # Tối ưu hóa cho Input 32 (CIFAR-10 chuẩn):
        #   Giảm stride cho một số block và stem để feature size hợp lý (8x8) trước AvgPool.
        if size_image == 32:
            self.stem_stride = 1 # Conv1: size 32
            self.b2_stride = 1   # Block 2: size 32 -> 32 (thay đổi từ s=2 thành s=1)
            self.b5_stride = 2   # Block 5: size 32 -> 16 (giữ nguyên s=2)
            self.b8_stride = 1   # Block 8: size 16 -> 16 (thay đổi từ s=2 thành s=1)
            self.b9_stride = 2   # Block 9: size 16 -> 8  (giữ nguyên s=2)
            # Feature size cuối cùng: 32 (stem) -> 16 (B5 s=2) -> 8 (B9 s=2). 8x8.
        else: # Mặc định hoặc 224
            # Sơ đồ: Stem S=2 (size 112), Blocks 2, 5, 8, 9 S=2
            self.stem_stride = 2 # Conv1: size 112
            self.b2_stride = 2   # Block 2: size 112 -> 56
            self.b5_stride = 2   # Block 5: size 56 -> 28
            self.b8_stride = 2   # Block 8: size 28 -> 14
            self.b9_stride = 2   # Block 9: size 14 -> 7
            # Feature size cuối cùng: 224 (stem s=2) -> 112 -> 56 -> 28 -> 14 -> 7. 7x7.

        # ───────────────── Cấu trúc mạng ─────────────────
        # STEM
        # Sơ đồ: I=224, C=3 -> Conv k=3x3, S=stride_chosen, P=1, out_channels=32
        # Tôi sẽ loại bỏ F.relu cho Conv đầu tiên trong forward pass vì stem conv này là linear. Pw1 của block sẽ xử lý.
        #self.conv1 = nn.Conv2d(in_channels = 3, out_channels=32, kernel_size=3, stride=self.stem_stride, padding=1)

        # Stages and Blocks
        # Ghi chú in-channels và out-channels khớp sơ đồ BT3 blocks:
        self.BlockBT3_1 = BlockBT3(in_channels = 32, out_channels=64, s=1)
        # Block 2 stride tùy thuộc vào kích thước input
        self.BlockBT3_2 = BlockBT3(in_channels = 64, out_channels=64, s=self.b2_stride)
        self.BlockBT3_3 = BlockBT3(in_channels = 64, out_channels=128, s=1)
        self.BlockBT3_4 = BlockBT3(in_channels = 128, out_channels=128, s=1)
        # Block 5 stride 2 (có thể hạ)
        self.BlockBT3_5 = BlockBT3(in_channels = 128, out_channels=256, s=self.b5_stride)
        self.BlockBT3_6 = BlockBT3(in_channels = 256, out_channels=256, s=1)
        self.BlockBT3_7 = BlockBT3(in_channels = 256, out_channels=256, s=1)
        # Block 8 stride s8_stride
        self.BlockBT3_8 = BlockBT3(in_channels = 256, out_channels=512, s=self.b8_stride)
        # Block 9 stride s9_stride
        self.BlockBT3_9 = BlockBT3(in_channels = 512, out_channels=512, s=self.b9_stride)

        # Final Conv k=1x1, S=1, P=0, in_channels=512, out_channels=1024
        self.conv2 = nn.Conv2d(in_channels = 512, out_channels=1024, kernel_size=1, stride=1, padding=0)

        # AvgPool K=whole, S=1 (AdaptiveAvgPool2d tự xử lý 7x7 hoặc 8x8)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        # FC
        # Ghi chú: K=whole, 1024. FC(1024, n_class)
        self.fc = nn.Linear(1024, n_class)

    def forward(self, x):
        # STEM
        x = self.conv1(x) # Output size (32, 112, 112) cho 224 hoặc (32, 32, 32) cho 32. Linear.
        
        # Stages
        x = self.BlockBT3_1(x)
        x = self.BlockBT3_2(x)
        x = self.BlockBT3_3(x)
        x = self.BlockBT3_4(x)
        x = self.BlockBT3_5(x)
        x = self.BlockBT3_6(x)
        x = self.BlockBT3_7(x)
        x = self.BlockBT3_8(x)
        x = self.BlockBT3_9(x)
        
        # Final layers
        # Ghi chú: F.relu(self.conv2(x)) trong NetBT3 forward pass. Conv2 1x1. Gốc code có. Giữ F.relu như gốc để fidelity.
        x = F.relu(self.conv2(x)) 
        x = self.avgpool(x)
        # Tối ưu hóa: Flatten feature vector 1024x1x1 -> 1024. `torch.flatten(x, 1)` tốt hơn `view`.
        x = torch.flatten(x, 1) 
        
        # prints... removing them in file content provided to user.
        
        # prints removed from code block, keep prints outside or in separate test for users to see.
        
        x = self.fc(x)
        return x