# 🚦 Vietnamese License Plate Detection & OCR

Một hệ thống AI nhận diện và trích xuất ký tự trên biển số xe đường phố Việt Nam, hoạt động hiệu quả với cả Biển Số Dài (BSD) và Biển Số Vuông (BSV).

## 📸 Demo Giao Diện

## 🌟 Tính năng nổi bật
* **Phát hiện biển số cực nhạy:** Sử dụng mạng nơ-ron YOLOv8 được Transfer Learning trực tiếp trên bộ dữ liệu đường phố thực tế tại Việt Nam, bắt tốt các góc nghiêng và khoảng cách xa.
* **Phân loại chính xác:** Tự động định vị, cắt ảnh (Crop) và dán nhãn Bounding Box phân biệt rõ kiểu dáng biển số.
* **Trích xuất ký tự trung thực:** Tích hợp EasyOCR kết hợp xử lý ảnh OpenCV. Hệ thống giữ nguyên văn bản thô (Raw text) kèm theo **Độ tự tin (Confidence Score)** để tối ưu trải nghiệm người dùng, giúp đánh giá chính xác độ tin cậy của thuật toán trên từng khung hình.
* **Giao diện thân thiện:** Tương tác trực tiếp, upload ảnh và hiển thị kết quả phân tách rõ ràng qua nền tảng web ứng dụng Streamlit.

## 🛠️ Công nghệ sử dụng
* **Computer Vision:** YOLOv8 (Ultralytics), OpenCV
* **OCR:** EasyOCR
* **Web Framework:** Streamlit
* **Huấn luyện mô hình:** Google Colab (GPU T4)

## ⚠️ Hạn chế & Hướng phát triển (Limitations & Future Work)
- **Hạn chế của OCR đại trà:** Hệ thống hiện tại trích xuất cực tốt trên biển số nền trắng. Tuy nhiên, thư viện EasyOCR gặp khó khăn vật lý với **biển số nền vàng** (xe khách, xe tải) ở điều kiện lóa sáng hoặc mờ. Ngoài ra, phông chữ đặc thù của biển số Việt Nam (ví dụ: chữ `B` và số `8`, số `3` và chữ `i`) dễ gây nhầm lẫn cho các mô hình OCR đọc văn bản thông thường.
- **Hướng khắc phục tương lai:** Thu thập thêm dữ liệu crop biển số và huấn luyện một mô hình OCR chuyên biệt (ví dụ: kiến trúc LPRNet hoặc CRNN) học thuật toán nhận diện chuẩn xác 36 ký tự phông chữ giao thông Việt Nam.

## 🚀 Hướng dẫn chạy thử trên máy cá nhân
1. Clone repository này về máy và di chuyển vào thư mục dự án.
2. Cài đặt các thư viện cần thiết:
   pip install -r requirements.txt
3. Khởi chạy trạm nhận diện AI trên trình duyệt:
   streamlit run app.py
