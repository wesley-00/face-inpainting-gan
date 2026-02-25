# face-inpainting-gan
# High-Res Face Inpainting

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

Dự án này triển khai một hệ thống Deep Learning sử dụng mạng GAN (Generative Adversarial Network) để giải quyết bài toán khôi phục ảnh khuôn mặt (Face Inpainting). Hệ thống có khả năng tự động tái tạo các vùng bị hỏng, bị xước hoặc bị che khuất trên khuôn mặt người dùng một cách chân thực nhất.

Đặc biệt, dự án tích hợp thuật toán **High-Res Blending**, cho phép người dùng đưa vào bức ảnh gốc chất lượng cao (Full HD / 4K), AI sẽ xử lý và ghép nối phần phục hồi một cách mượt mà mà không làm giảm độ phân giải của toàn bức ảnh.

---

## Kiến trúc Hệ thống



[Image of Generative Adversarial Network architecture]


Hệ thống được xây dựng với các kỹ thuật tiên tiến nhất trong lĩnh vực Computer Vision:

* **Generator (U-Net + Self-Attention):** Trái tim của hệ thống là mạng U-Net kết hợp với module Self-Attention. Cơ chế Attention giúp mô hình hiểu được ngữ cảnh toàn cục của khuôn mặt (ví dụ: tạo ra con mắt trái đồng bộ với con mắt phải) thay vì chỉ nhìn vào các điểm ảnh lân cận.
* **Discriminator (PatchGAN):** Đánh giá tính chân thực của bức ảnh trên từng vùng nhỏ (Patch) thay vì toàn bộ ảnh, giúp tái tạo chi tiết lỗ chân lông và kết cấu da tự nhiên hơn.
* **Kỹ thuật ổn định:** Sử dụng **Spectral Normalization** và **Instance Noise** để chống lại hiện tượng Mode Collapse, giúp hai mạng G và D huấn luyện cân bằng.
* **Hàm Loss Đa mục tiêu:** Tối ưu hóa mô hình bằng sự kết hợp của L1 Loss (cấu trúc hình học), VGG Perceptual Loss (đặc trưng ngữ nghĩa) và Adversarial Loss (tính chân thực).

---

## Cấu trúc Thư mục

```text
face-inpainting-gan/
├── data/                  # Nơi chứa dataset (VD: CelebA-HQ)
├── checkpoints/           # Thư mục chứa trọng số mô hình đã huấn luyện (.pth)
├── test_samples/          # Thư mục chứa ảnh mẫu dùng cho UI
├── outputs/               # Nơi lưu báo cáo WandB, ảnh sinh ra lúc train
│
├── config.py              # File cấu hình trung tâm (Hyperparameters, Paths)
├── dataset.py             # Pipeline xử lý dữ liệu (DataLoader, tạo Mask tự động)
├── models.py              # Kiến trúc mạng Generator, Discriminator, Attention
├── loss.py                # Định nghĩa các hàm tính Loss
├── utils.py               # Các hàm hỗ trợ (Metrics PSNR/SSIM, lưu checkpoint)
│
├── train.py               # Script huấn luyện mô hình từ đầu
├── evaluate.py            # Script đo lường độ chính xác của mô hình
├── app.py                 # Giao diện Web tương tác (Gradio)
└── requirements.txt       # Danh sách thư viện phụ thuộc
```
Hướng dẫn Cài đặt

Yêu cầu hệ thống: Python 3.9+ và phần cứng có GPU (NVIDIA) hỗ trợ CUDA.

Bước 1: Clone kho mã nguồn về máy tính
Bash

git clone [https://github.com/TÊN_TÀI_KHOẢN_CỦA_BẠN/face-inpainting-gan.git](https://github.com/TÊN_TÀI_KHOẢN_CỦA_BẠN/face-inpainting-gan.git)
cd face-inpainting-gan

Bước 2: Cài đặt các thư viện cần thiết
Bash

pip install -r requirements.txt

Bước 3: Tải file trọng số (Model Weights)

    Tạo thư mục checkpoints/ tại thư mục gốc của dự án.

    Đặt file trọng số tốt nhất của bạn (ví dụ: best_model.pth) vào thư mục checkpoints/.

🚀 Hướng dẫn Sử dụng
1. Khởi chạy Ứng dụng Web (Inference)

Dự án cung cấp một giao diện web trực quan giúp bạn dễ dàng thử nghiệm mô hình mà không cần gõ lệnh phức tạp.
Bash

python app.py

    Mở trình duyệt và truy cập vào địa chỉ mạng cục bộ (ví dụ: http://127.0.0.1:7860).

    Tải một bức ảnh khuôn mặt lên, dùng cọ vẽ bôi đen vùng bị hỏng. Hệ thống sẽ tự động khoanh vùng Bounding Box và phục hồi vùng đó.

2. Huấn luyện Mô hình (Training)

Để huấn luyện lại mô hình với dữ liệu của bạn:

    Chuẩn bị bộ dữ liệu ảnh khuôn mặt (ví dụ: CelebA-HQ) và đặt vào thư mục data/.

    Mở file config.py để tinh chỉnh các siêu tham số (Batch size, Learning rate, Epochs...).

    Chạy script huấn luyện:

Bash

python train.py

(Quá trình huấn luyện tự động tích hợp Weights & Biases (WandB) để theo dõi đồ thị Loss, PSNR theo thời gian thực).
3. Đánh giá Mô hình (Evaluation)

Để kiểm tra các chỉ số đo lường học thuật trên tập Validation:
Bash

python evaluate.py

Script sẽ tính toán và in ra các chỉ số: PSNR, SSIM, LPIPS.
📊 Kết quả Huấn luyện

Chi tiết về quá trình hội tụ của mô hình và các chỉ số đánh giá đã được trích xuất thành tệp báo cáo PDF (có thể xem trong phần file đính kèm của repository). Hệ thống vượt qua mức Baseline tiêu chuẩn với khả năng kết xuất hình ảnh sắc nét, không để lại vết ghép (seam) rõ rệt.
