import os
import tensorflow as tf
from keras.applications import MobileNetV3Large
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import logging

# ========== Cấu hình ==========
KICH_THUOC_ANH = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
DATASET_PATH = './dataset_3_nhom_split'

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== Data Generator ==========
def tao_data_generators(duong_dan_du_lieu):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(duong_dan_du_lieu, 'train'),
        target_size=(KICH_THUOC_ANH, KICH_THUOC_ANH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(duong_dan_du_lieu, 'validation'),
        target_size=(KICH_THUOC_ANH, KICH_THUOC_ANH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(duong_dan_du_lieu, 'test'),
        target_size=(KICH_THUOC_ANH, KICH_THUOC_ANH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    return train_gen, val_gen, test_gen

# ========== Xây dựng mô hình ==========
def xay_dung_mo_hinh(so_lop):
    base_model = MobileNetV3Large(weights='imagenet', include_top=False,
                                  input_shape=(KICH_THUOC_ANH, KICH_THUOC_ANH, 3))
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(so_lop, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def fine_tune_model(model, base_model, so_lop=30):
    base_model.trainable = True
    for layer in base_model.layers[:-so_lop]:
        layer.trainable = False
    model.compile(optimizer=Adam(LEARNING_RATE / 20),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ========== Callbacks ==========
def tao_callbacks():
    return [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3,
                          min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    ]

# ========== Vẽ biểu đồ ==========
def ve_bieu_do(history, history_ft):
    acc = history.history['accuracy'] + history_ft.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_ft.history['val_accuracy']
    loss = history.history['loss'] + history_ft.history['loss']
    val_loss = history.history['val_loss'] + history_ft.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.legend()
    plt.title('Biểu đồ Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.legend()
    plt.title('Biểu đồ Loss')
    plt.tight_layout()
    plt.show()

# ========== Ma trận nhầm lẫn ==========
def hien_thi_ma_tran_nham_lan(model, test_gen):
    y_pred_probs = model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes
    labels = list(test_gen.class_indices.keys())

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Ma trận nhầm lẫn')
    plt.tight_layout()
    plt.show()

    print("\nBáo cáo chi tiết:")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

# ========== Huấn luyện ==========
def train():
    logger.info("\n🚀 Bắt đầu huấn luyện...")
    train_gen, val_gen, test_gen = tao_data_generators(DATASET_PATH)
    so_lop = len(train_gen.class_indices)
    logger.info(f"Số lớp: {so_lop} | {train_gen.class_indices}")

    # ====== Tính class_weight tự động ======
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    y_train = train_gen.classes
    class_labels = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=class_labels, y=y_train)
    indexed_weight = dict(zip(class_labels, weights))

    logger.info(f"\n⚖️ class_weight sẽ dùng trong huấn luyện: {indexed_weight}")

    # ====== Huấn luyện mô hình ======
    model, base_model = xay_dung_mo_hinh(so_lop)
    model.compile(optimizer=Adam(LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS // 2,
        callbacks=tao_callbacks(),
        class_weight=indexed_weight  # ✅ THÊM VÀO ĐÂY
    )

    model = fine_tune_model(model, base_model)
    history_ft = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS // 2,
        callbacks=tao_callbacks(),
        class_weight=indexed_weight  # ✅ VÀ ĐÂY NỮA
    )

    model.save("trashnet_model.keras", save_format="keras")
    logger.info("\n✅ Đã lưu mô hình tại trashnet_model.keras")

    ve_bieu_do(history, history_ft)

    logger.info("\n📊 Đánh giá mô hình trên tập test...")
    loss, accuracy = model.evaluate(test_gen, verbose=1)
    logger.info(f"\n🎯 Kết quả test set: Accuracy = {accuracy:.4f}, Loss = {loss:.4f}")
    print(f"\n✅ Đánh giá hoàn tất. Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

    hien_thi_ma_tran_nham_lan(model, test_gen)


# ========== Entry Point ==========
if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    train()
    print("\n🎉 Huấn luyện hoàn tất và mô hình đã được lưu.")
