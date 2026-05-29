import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import easyocr

# --- HÀM TIỀN XỬ LÝ ẢNH (TĂNG ĐỘ TƯƠNG PHẢN & LÀM NÉT) ---
def preprocess_plate(img_array):
    # 1. Chuyển sang ảnh xám (Grayscale)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 2. Phóng to ảnh để OCR dễ đếm pixel hơn
    enlarged = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    
    # 3. Cân bằng ánh sáng (CLAHE) - Cứu tinh cho ảnh lóa/mờ
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(enlarged)
    
    # EasyOCR nhận ảnh màu RGB nên ta chuyển ngược lại
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

# --- HÀM LẤY TEXT THÔ VÀ TÍNH ĐỘ TỰ TIN ---
def get_raw_ocr_result(ocr_result):
    if len(ocr_result) == 0:
        return "", 0.0
        
    texts = [detection[1] for detection in ocr_result]
    confs = [detection[2] for detection in ocr_result]
    
    # ĐỊNH DẠNG TEXT
    if len(texts) == 1:
        # Trường hợp 1 dòng (Biển dài): Thay khoảng trắng bằng dấu gạch ngang
        full_raw_text = texts[0].replace(" ", "-")
    else:
        # Trường hợp nhiều dòng (Biển vuông): Nối với nhau bằng dấu |
        full_raw_text = " | ".join(texts)
        
    avg_conf = sum(confs) / len(confs)
        
    return full_raw_text, avg_conf

@st.cache_resource
def load_yolo():
    return YOLO('best.pt')

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False) 

model = load_yolo()
reader = load_ocr()

st.title("Nhận Diện Biển Số Xe AI (Pro Vision)")

uploaded_file = st.file_uploader("Tải ảnh xe cộ lên...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh Gốc', use_container_width=True)
    
    if st.button("Phân tích ngay!"):
        with st.spinner("Đang xử lý ảnh và đọc văn bản..."):
            
            results = model(image, imgsz=1024, conf=0.15)
            
            res_plotted = results[0].plot()
            res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            st.image(res_plotted, caption='Kết Quả Nhận Diện Khung', use_container_width=True)
            
            st.subheader("Chi tiết đọc được:")
            
            img_array = np.array(image)
            if img_array.shape[-1] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                
            boxes = results[0].boxes
            
            if len(boxes) == 0:
                st.warning("Không tìm thấy biển số nào trong ảnh!")
            else:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # 1. Cắt ảnh nguyên bản
                    cropped_plate = img_array[y1:y2, x1:x2]
                    
                    # 2. Bơm ma thuật Tiền Xử Lý vào!
                    processed_plate = preprocess_plate(cropped_plate)
                    
                    # 3. Đưa ảnh đã làm nét cho EasyOCR
                    ocr_result = reader.readtext(processed_plate)
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Hiện luôn cái ảnh trắng đen tương phản cao cho ngầu
                        st.image(processed_plate, caption=f"Mục tiêu {i+1} (Đã xử lý nét)", use_container_width=True)

                    with col2:
                        if len(ocr_result) == 0:
                            st.error("❌ Ảnh quá mờ hoặc lóa sáng, hệ thống OCR không thể đọc!")
                        else:
                            raw_text, avg_conf = get_raw_ocr_result(ocr_result)
                            st.success(f"✅ BIỂN SỐ: {raw_text} (Độ tự tin: {avg_conf:.2f})")
                    
                    st.divider()