import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"        # Tắt ONEDNN delegate
os.environ["TF_DELEGATE_ENABLE"] = "0"           # Tắt XNNPACK
os.environ["TF_USE_LEGACY_KERAS"] = "1"          # ⚠️ Ép dùng Keras chuẩn, không TFLite hóa ngầm

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageOps
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import hashlib
from datetime import datetime
import io
import base64
import cv2
import json

# Tắt các warning của TensorFlow
tf.get_logger().setLevel('ERROR')
import warnings
warnings.filterwarnings('ignore')

# Đảm bảo set_page_config là dòng đầu tiên
st.set_page_config(
    page_title="Phân loại rác thông minh AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS ===
def load_custom_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #00C9FF;
    }
    
    .drag-drop-area {
        border: 2px dashed #00C9FF;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
    }
    
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #f39c12;
    }
    
    .recycling-tip {
        background: #e8f5e8;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    </style>
    """, unsafe_allow_html=True)

# === Thông tin lớp rác với tips tái chế ===
LABELS_INFO = {
    "huu_co": {
        "vi": "Rác hữu cơ",
        "info": "Thức ăn thừa, rau củ quả, lá cây...",
        "image": "resources/huu_co.jpg",
        "recycling_tips": [
            "Ủ phân để tạo phân bón hữu cơ",
            "Không trộn với rác tái chế",
            "Sử dụng thùng ủ phân tại nhà",
            "Giảm lãng phí thực phẩm"
        ],
        "collection_points": "Thùng rác hữu cơ hoặc khu ủ phân",
        "color": "#006400"
    },
    "tai_che": {
        "vi": "Rác tái chế",
        "info": "Giấy, nhựa, kim loại, thủy tinh, vải, đồ da, pin...",
        "image": "resources/tai_che.jpg",
        "recycling_tips": [
            "Phân loại đúng trước khi bỏ vào thùng tái chế",
            "Rửa sạch nhựa, kim loại, thủy tinh",
            "Mang pin đến điểm thu gom chuyên dụng",
            "Tái chế tiết kiệm tài nguyên"
        ],
        "collection_points": "Thùng tái chế (xanh lá, xanh dương, xám, vàng)",
        "color": "#28a745"
    },
    "vo_co": {
        "vi": "Rác vô cơ",
        "info": "Túi nilon, xốp, ly nhựa dùng một lần...",
        "image": "resources/vo_co.jpg",
        "recycling_tips": [
            "Giảm sử dụng nhựa dùng một lần",
            "Không trộn với rác tái chế",
            "Xử lý đúng cách để tránh ô nhiễm"
        ],
        "collection_points": "Thùng rác thông thường",
        "color": "#800080"
    }
}

# === Khởi tạo và tải dữ liệu lịch sử ===
HISTORY_FILE = "history_data.json"

def load_history():
    """Tải lịch sử từ file"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except:
        return []

def save_history(history_data):
    """Lưu lịch sử vào file"""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Lỗi khi lưu lịch sử: {e}")

# === Initialize session state ===
if 'history_data' not in st.session_state:
    st.session_state.history_data = load_history()
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = {}
if 'camera_enabled' not in st.session_state:
    st.session_state.camera_enabled = False

@st.cache_resource
def load_model():
    try:
        # Load mô hình kiểu Keras thuần
        from keras.models import load_model
        return load_model("trashnet_model.keras", compile=False)
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        st.stop()

model = load_model()
class_names = list(LABELS_INFO.keys())

# === Utility Functions ===
def get_image_hash(image):
    """Tạo hash cho ảnh để phát hiện trùng lặp"""
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    return hashlib.md5(image_bytes.getvalue()).hexdigest()

def auto_rotate_image(image):
    """Tự động xoay ảnh dựa trên dữ liệu EXIF"""
    try:
        return ImageOps.exif_transpose(image)
    except:
        return image

def process_image(image: Image.Image):
    """Xử lý ảnh"""
    # Tự động xoay
    image = auto_rotate_image(image)
    
    img = image.convert("RGB").resize((224, 224))
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = array / 255.0
    return np.expand_dims(array, axis=0)

# === Main Interface ===
def main():
    load_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Phân loại rác thông minh</h1>
        <p>Ứng dụng AI tiên tiến giúp phân loại rác và bảo vệ môi trường</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cài đặt")
        
        # Camera toggle
        st.subheader("📷 Camera")
        camera_toggle = st.toggle("Bật camera", value=st.session_state.camera_enabled)
        st.session_state.camera_enabled = camera_toggle
        
        # Clear history
        if st.button("🗑️ Xóa lịch sử"):
            st.session_state.history_data = []
            st.session_state.processed_images = {}
            save_history([])
            st.success("Đã xóa lịch sử!")
            st.rerun()
    
    # Educational section - moved up
    with st.expander("📚 Tìm hiểu về phân loại rác"):
        st.markdown("### 🌍 Tại sao phân loại rác quan trọng?")
        st.markdown("""
        - **Bảo vệ môi trường**: Giảm ô nhiễm đất, nước và không khí
        - **Tiết kiệm tài nguyên**: Tái chế giúp tiết kiệm nguyên liệu thô
        - **Giảm khí thải**: Giảm phát thải CO2 và khí nhà kính
        - **Kinh tế tuần hoàn**: Tạo ra giá trị từ chất thải
        """)
        
        st.markdown("### ♻️ Hướng dẫn chi tiết từng loại rác:")
        
        for label, info in LABELS_INFO.items():
            with st.container():
                st.markdown(f"#### {info['vi']}")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if os.path.exists(info['image']):
                        st.image(info['image'], caption=f"Hình minh họa: {info['vi']}", width=150)
                
                with col2:
                    st.markdown(f"**Mô tả:** {info['info']}")
                    st.markdown(f"**Điểm thu gom:** {info['collection_points']}")
                    
                    st.markdown("**Cách tái chế:**")
                    for tip in info['recycling_tips']:
                        st.markdown(f"• {tip}")
                
                st.markdown("---")
    
    # Main content
    st.subheader("📤 Tải lên hình ảnh")
    
    # Drag and drop area
    st.markdown("""
    <div class="drag-drop-area">
        <h3>🎯 Kéo thả hoặc click để tải ảnh</h3>
        <p>Hỗ trợ JPG, PNG, JPEG • Tối đa 200MB mỗi ảnh</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Chọn nhiều ảnh", 
        accept_multiple_files=True, 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    # Camera input (only if enabled)
    camera_image = None
    if st.session_state.camera_enabled:
        camera_image = st.camera_input("Chụp ảnh rác")
        if camera_image:
            uploaded_files = [camera_image] + (uploaded_files or [])
    
    # Process uploaded images
    if uploaded_files:
        st.subheader("🔍 Kết quả phân loại")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Đang xử lý ảnh {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            # Load and process image
            img = Image.open(uploaded_file)
            img_hash = get_image_hash(img)
            
            # Process image
            input_tensor = process_image(img)
            
            # Predict
            with st.spinner("Đang phân tích..."):
                prediction = model.predict(input_tensor, verbose=0)[0]
                predicted_idx = np.argmax(prediction)
                predicted_label = class_names[predicted_idx]
                confidence = prediction[predicted_idx]
            
            # Store result
            vi_label = LABELS_INFO[predicted_label]["vi"]
            description = LABELS_INFO[predicted_label]["info"]
            
            result = {
                'file': uploaded_file,
                'image': img,
                'label': predicted_label,
                'vi_label': vi_label,
                'confidence': confidence,
                'description': description,
                'hash': img_hash
            }
            results.append(result)
            
            # Store in session
            st.session_state.processed_images[img_hash] = result
            
            # Thêm vào lịch sử
            history_entry = {
                "Tên ảnh": uploaded_file.name,
                "Loại rác": vi_label,
                "Độ chính xác (%)": f"{confidence*100:.2f}%",
                "Thời gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.history_data.append(history_entry)
        
        # Lưu lịch sử vào file
        save_history(st.session_state.history_data)
        
        # Clear progress
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        for result in results:
            with st.container():
                st.markdown(f"""
                <div class="result-card">
                    <h4>📋 Kết quả: {result['file'].name}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(result['image'], caption="Ảnh gốc", use_container_width=True)
                
                with col2:
                    st.markdown(f"**🏷️ Loại rác:** `{result['vi_label']}`")
                    st.markdown(f"**📝 Mô tả:** {result['description']}")
                
                # Recycling tips
                st.markdown(f"""
                <div class="recycling-tip">
                    <h5>♻️ Hướng dẫn tái chế {result['vi_label']}:</h5>
                </div>
                """, unsafe_allow_html=True)
                
                tips = LABELS_INFO[result['label']]['recycling_tips']
                for tip in tips:
                    st.markdown(f"• {tip}")
                
                collection_point = LABELS_INFO[result['label']]['collection_points']
                st.info(f"📍 **Điểm thu gom:** {collection_point}")
                
                st.markdown("---")
    
    # Analytics Dashboard
    if st.session_state.history_data:
        st.subheader("📈 Phân tích thống kê")
        
        df = pd.DataFrame(st.session_state.history_data)
        
        # Tạo cột cho tháng để phân tích theo thời gian
        df['Thời gian'] = pd.to_datetime(df['Thời gian'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution chart
            trash_counts = df['Loại rác'].value_counts()
            colors = [LABELS_INFO[label]['color'] for label in LABELS_INFO.keys() if LABELS_INFO[label]['vi'] in trash_counts.index]
            
            fig = px.pie(
                values=trash_counts.values,
                names=trash_counts.index,
                title="Phân bố loại rác",
                color_discrete_sequence=colors
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Recycling tips for each type
            st.markdown("### ♻️ Hướng dẫn tái chế")
            
            for label, info in LABELS_INFO.items():
                if info['vi'] in trash_counts.index:
                    count = trash_counts[info['vi']]
                    st.markdown(f"**{info['vi']} ({count} ảnh):**")
                    st.markdown(f"📍 {info['collection_points']}")
                    
                    # Show first 2 recycling tips
                    for i, tip in enumerate(info['recycling_tips'][:2]):
                        st.markdown(f"• {tip}")
                    
                    st.markdown("---")
        
        # ĐÃ XÓA: Heatmap by day of week and hour
        # Phần code heatmap hoạt động theo giờ và ngày đã được gỡ bỏ hoàn toàn
        
        # Top statistics
        st.subheader("📊 Thống kê tổng quan")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tổng số ảnh", len(df))
        
        with col2:
            most_common = df['Loại rác'].mode().iloc[0] if not df.empty else "Chưa có"
            st.metric("Loại rác phổ biến", most_common)
        
        with col3:
            # Tính độ chính xác trung bình
            df['Độ chính xác số'] = df['Độ chính xác (%)'].str.rstrip('%').astype(float)
            avg_accuracy = df['Độ chính xác số'].mean()
            st.metric("Độ chính xác TB", f"{avg_accuracy:.1f}%")
        
        with col4:
            # Số loại rác khác nhau
            unique_types = df['Loại rác'].nunique()
            st.metric("Loại rác đã phân loại", unique_types)
        
        # Export options
        st.subheader("📤 Xuất dữ liệu")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📊 Tải tệp CSV",
                data=csv,
                file_name=f"phan_loai_rac_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Phân loại rác')
            
            st.download_button(
                "📈 Tải tệp Excel",
                data=excel_buffer.getvalue(),
                file_name=f"phan_loai_rac_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            # Backup history
            history_json = json.dumps(st.session_state.history_data, ensure_ascii=False, indent=2)
            st.download_button(
                "💾 Backup lịch sử",
                data=history_json,
                file_name=f"backup_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # History table
        st.subheader("📝 Lịch sử chi tiết")
        
        # Search and filter
        search_term = st.text_input("🔍 Tìm kiếm trong lịch sử:")
        if search_term:
            filtered_df = df[df['Tên ảnh'].str.contains(search_term, case=False, na=False)]
        else:
            filtered_df = df
        
        # Display with formatting
        display_df = filtered_df[['Tên ảnh', 'Loại rác', 'Độ chính xác (%)', 'Thời gian']].copy()
        
        st.dataframe(
            display_df,
            use_container_width=True
        )
    
    # Footer
    st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
    <h4>📬 Liên hệ</h4>
    <p>Nếu bạn có thắc mắc, góp ý hoặc muốn hợp tác, vui lòng liên hệ:</p>
    <p>📧 Email: <a href="mailto:fftt0519@gmail.com">fftt0519@gmail.com</a></p>
    <p>📞 Điện thoại: 0339336571</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
