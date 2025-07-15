import os
import logging
from PIL import Image
import numpy as np

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Danh sách nhãn
LABELS = [
    'battery', 'biological', 'cardboard', 'clothes',
    'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash'
]

def kiem_tra_anh(img_path):
    """Kiểm tra ảnh hợp lệ và không bị đen toàn bộ"""
    try:
        img = Image.open(img_path)
        img.verify()  # Kiểm tra file ảnh hỏng
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        if img_array.max() == 0:
            logger.warning(f"Ảnh toàn đen: {img_path}")
            return False
        return True
    except Exception as e:
        logger.warning(f"Không thể đọc ảnh {img_path}: {e}")
        return False

def resize_anh(img_path, output_path, size=(224, 224)):
    """Resize ảnh và lưu sang đuôi .jpg"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(size, Image.Resampling.LANCZOS)
        img.save(output_path, format='JPEG')
        return True
    except Exception as e:
        logger.warning(f"Lỗi resize ảnh {img_path}: {e}")
        return False

def lam_sach_dataset(input_dir, output_dir="dataset_cleaned"):
    """Làm sạch toàn bộ dataset và resize ảnh"""
    os.makedirs(output_dir, exist_ok=True)

    for label in LABELS:
        input_path = os.path.join(input_dir, label)
        output_path = os.path.join(output_dir, label)
        os.makedirs(output_path, exist_ok=True)

        if not os.path.exists(input_path):
            logger.warning(f"Không tìm thấy thư mục: {input_path}")
            continue

        files = os.listdir(input_path)
        tong = len(files)
        thanh_cong = 0

        for file_name in files:
            src_path = os.path.join(input_path, file_name)

            if not kiem_tra_anh(src_path):
                continue

            new_name = os.path.splitext(file_name)[0] + "_224x224.jpg"
            dst_path = os.path.join(output_path, new_name)

            if resize_anh(src_path, dst_path):
                thanh_cong += 1

        logger.info(f"{label}: {thanh_cong}/{tong} ảnh đã xử lý thành công")

    logger.info("✅ Đã hoàn tất làm sạch và resize dữ liệu!")

if __name__ == "__main__":
    INPUT_DIR = "garbage_classification"
    lam_sach_dataset(INPUT_DIR)
