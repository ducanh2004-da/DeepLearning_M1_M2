# 🖼️ Image Classification Project - CNN Models

**Tác giả:** Đỗ Đức Anh, Lã Huy Hoàng, Lê Huy Hoàng  
**Trường:** Đại học Công nghệ Kỹ thuật TP.HCM (HCMUTE)  

Dự án này bao gồm hai giai đoạn chính (Giữa kỳ và Cuối kỳ), tập trung vào việc xây dựng, huấn luyện và đánh giá các mô hình Mạng nơ-ron tích chập (CNN) để phân loại hình ảnh trên nhiều tập dữ liệu khác nhau.

---

## 📂 Cấu trúc Thư mục & Chức năng

### 1. 📁 Folder `M1` (Dự án Giữa kỳ)
Tập trung vào phân loại 5 lớp động vật với hai độ phân giải khác nhau.
- **Dữ liệu:** 5 class động vật (Ảnh kích thước `32x32` và `224x224`). *(Lưu ý: Dữ liệu đã được tích hợp sẵn, không cần chạy script tải data).*
- **Huấn luyện:** Mô hình M1 được train và test trong **100 epochs**.

### 2. 📁 Folder `M2` (Dự án Cuối kỳ)
Mở rộng dự án với các tập dữ liệu chuẩn và phức tạp hơn, đi kèm giao diện Web trực quan.
- **Dữ liệu:** CIFAR-10 và CIFAR-100.
- **Huấn luyện:** Mô hình M2 được train và test trong **200 epochs**.

### 📜 Các File Thực thi Chính
| Tên File | Vị trí | Chức năng |
| :--- | :--- | :--- |
| `splitdata.py` | Gốc M1 | Tải và chia tập dữ liệu thành Train/Test cho các mô hình cơ bản. |
| `splitdata_Cifar.py` | Gốc M1 | Tải và chia tập dữ liệu riêng cho CIFAR-10 và CIFAR-100. |
| `train.py` | Gốc M1 | Chạy quá trình huấn luyện và kiểm thử (Train & Test) cho mô hình giữa kỳ (100 epochs). |
| `train_Cifar.py` | `M2/` | Chạy quá trình huấn luyện và kiểm thử (Train & Test) cho mô hình cuối kỳ trên CIFAR (200 epochs). |
| `app.py` | `M2/` | Triển khai Web App. Cho phép người dùng tải lên một hình ảnh bất kỳ để mô hình dự đoán và trả về nhãn (class) tương ứng, chạy với lệnh streamlit run ..<đường dẫn app>... |

---

## 🚀 Hướng dẫn Sử dụng (How to Run)

### Bước 1: Chuẩn bị Dữ liệu (Data Preparation)
*Lưu ý: Bỏ qua bước này nếu bạn chỉ muốn chạy lại mô hình M1 (dữ liệu M1 đã có sẵn).*
Để tải và phân chia dữ liệu cho mô hình M2 (CIFAR), hãy chạy lệnh:
```bash
python splitdata_Cifar.py
