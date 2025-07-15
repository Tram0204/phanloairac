import os
import shutil
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bản đồ nhóm: từ nhãn gốc -> nhóm mới
NHOM_RAC = {
    'biological': 'huu_co',
    'trash': 'vo_co',
    'shoes': 'vo_co',
    'battery': 'tai_che',
    'cardboard': 'tai_che',
    'glass': 'tai_che',
    'metal': 'tai_che',
    'paper': 'tai_che',
    'plastic': 'tai_che',
    'clothes': 'tai_che'
}

def gop_thanh_3_lop(input_dir="dataset_cleaned", output_dir="dataset_3_nhom"):
    """Gộp dữ liệu từ 10 lớp thành 3 lớp"""
    os.makedirs(output_dir, exist_ok=True)

    # Tạo các thư mục nhóm mới
    for group in set(NHOM_RAC.values()):
        os.makedirs(os.path.join(output_dir, group), exist_ok=True)

    # Duyệt qua từng thư mục gốc
    for label in NHOM_RAC:
        src_folder = os.path.join(input_dir, label)
        dst_group = NHOM_RAC[label]
        dst_folder = os.path.join(output_dir, dst_group)

        if not os.path.exists(src_folder):
            logger.warning(f"Không tìm thấy thư mục: {src_folder}")
            continue

        files = os.listdir(src_folder)
        for file in files:
            src_path = os.path.join(src_folder, file)
            dst_path = os.path.join(dst_folder, f"{label}_{file}")  # Thêm tiền tố nhãn cũ
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                logger.warning(f"Lỗi copy {src_path}: {e}")

        logger.info(f"✅ Đã gộp {len(files)} ảnh từ {label} vào nhóm {dst_group}")

    logger.info("🎉 Hoàn tất gộp dữ liệu thành 3 nhóm!")

if __name__ == "__main__":
    gop_thanh_3_lop()
