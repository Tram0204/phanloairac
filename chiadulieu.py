import os
import shutil
import random
from tqdm import tqdm

# Tỷ lệ chia
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Thư mục đầu vào và đầu ra
INPUT_DIR = 'dataset_3_nhom'
OUTPUT_DIR = 'dataset_3_nhom_split'

# Thiết lập seed để chia ngẫu nhiên nhưng có thể lặp lại
random.seed(42)

def tao_thu_muc(path):
    os.makedirs(path, exist_ok=True)

def chia_anh():
    # Tạo thư mục output
    for split in ['train', 'validation', 'test']:
        for class_name in os.listdir(INPUT_DIR):
            tao_thu_muc(os.path.join(OUTPUT_DIR, split, class_name))

    # Duyệt qua từng lớp
    for class_name in os.listdir(INPUT_DIR):
        class_path = os.path.join(INPUT_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        all_files = os.listdir(class_path)
        random.shuffle(all_files)

        total = len(all_files)
        val_count = int(total * VAL_RATIO)
        test_count = int(total * TEST_RATIO)
        train_count = total - val_count - test_count

        train_files = all_files[:train_count]
        val_files = all_files[train_count:train_count + val_count]
        test_files = all_files[train_count + val_count:]

        # Copy file vào từng thư mục tương ứng
        for file_set, split in zip([train_files, val_files, test_files], ['train', 'validation', 'test']):
            for file_name in tqdm(file_set, desc=f"📂 {class_name} → {split}", leave=False):
                src_path = os.path.join(class_path, file_name)
                dst_path = os.path.join(OUTPUT_DIR, split, class_name, file_name)
                shutil.copy2(src_path, dst_path)

    print("✅ Hoàn tất chia dữ liệu!")

if __name__ == "__main__":
    chia_anh()
