import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
import cv2
import os

from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.models import Sequential

import pathlib
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import ResNet50

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

data_dir = "/Users/bitterchoco/fx/image-classification/candle_data"

batch_size = 16
img_height = 300 
img_width = 300

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

class_weights = compute_class_weight(
#    class_weight='balanced',
    class_weight=None,
    classes=np.array(range(len(class_names))),
    y=np.concatenate([y.numpy() for _, y in train_ds])
)
class_weights = dict(enumerate(class_weights))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(2000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# チャート画像向け前処理関数
def chart_specific_preprocessing(image, label):
    # 画像をfloat32に正規化（0〜1）
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # コントラスト強調（ローソク足などの線をはっきりさせる）
    image = tf.image.adjust_contrast(image, 2.0)

    # per-image標準化（全体の明るさ/コントラスト差を吸収）
    image = tf.image.per_image_standardization(image)

    # ResNet50の事前学習モデルに合わせた前処理（0〜255に戻してからpreprocess_input）
    image = preprocess_input(image * 255.0)

    return image, label

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(factor=0.2),
    tf.keras.layers.RandomRotation(0.05),     # 少し回転
    tf.keras.layers.RandomTranslation(0.05, 0.05),  # 少し位置ずらす
    tf.keras.layers.GaussianNoise(0.02)
])

# チャート画像に特化した前処理適用
train_ds = train_ds.map(chart_specific_preprocessing, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(chart_specific_preprocessing, num_parallel_calls=AUTOTUNE)

num_classes = len(class_names)

img_path = "./candle_data/win/2025-2-28_1093.png"
img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
img = tf.keras.utils.img_to_array(img)
img = tf.image.convert_image_dtype(img, dtype=tf.float32)
img = tf.image.adjust_contrast(img, 2.0)
img = tf.image.per_image_standardization(img)
img = preprocess_input(img * 255.0)

# 可視化
plt.imshow((img + 1) / 2)
plt.axis("off")
plt.title("Preprocessed image for ResNet50")
plt.show()
# Functional API によるモデル構築（Grad-CAM対応）
base_model = ResNet50(input_shape=(img_height, img_width, 3),
                      include_top=False,
                      weights='imagenet')
base_model.trainable = True
last_conv_layer_name = "conv5_block3_out"
last_conv_layer = base_model.get_layer(last_conv_layer_name).output

x = layers.GlobalAveragePooling2D()(last_conv_layer)

x = layers.Dense(256, use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.4)(x)

x = layers.BatchNormalization()(x)  # Dense(128) の前
x = layers.Dense(128, use_bias=False)(x)
x = layers.BatchNormalization()(x)  # Dense(128) の後
x = layers.Activation('relu')(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

# モデル保存ディレクトリ
save_dir = "/Users/bitterchoco/fx/image-classification/saved_model"
os.makedirs(save_dir, exist_ok=True)

# コールバック設定
epochs = 100
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=6, min_lr=5e-5)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
checkpoint_cb = ModelCheckpoint(
    filepath=os.path.join(save_dir, "checkpoint_epoch_{epoch:02d}_valloss_{val_loss:.4f}.keras"),
    save_weights_only=False,
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[lr_scheduler, early_stopping, checkpoint_cb],
  class_weight=class_weights
)

# モデルを最終保存
model.save(os.path.join(save_dir, "resnet50_model.keras"))

# 履歴をCSV保存
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(save_dir, "training_history.csv"), index=False)

# プロット関数（再利用可能）
def plot_training_history(csv_path):
    history_df = pd.read_csv(csv_path)
    acc = history_df['accuracy']
    val_acc = history_df['val_accuracy']
    loss = history_df['loss']
    val_loss = history_df['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

# プロット実行
plot_training_history(os.path.join(save_dir, "training_history.csv"))

# -------- Grad-CAM 実装 --------
#def get_img_array(img_path, size):
#    img = tf.keras.utils.load_img(img_path, target_size=size)
#    array = tf.keras.utils.img_to_array(img)
#    array = np.expand_dims(array, axis=0)
#    return tf.keras.applications.resnet50.preprocess_input(array)
#
#def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
#    grad_model = tf.keras.models.Model(
#        [model.inputs], 
#        [model.get_layer(last_conv_layer_name).output, model.output]
#    )
#    with tf.GradientTape() as tape:
#        conv_outputs, predictions = grad_model(img_array)
#        if pred_index is None:
#            pred_index = tf.argmax(predictions[0])
#        class_channel = predictions[:, pred_index]
#
#    grads = tape.gradient(class_channel, conv_outputs)
#    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#
#    conv_outputs = conv_outputs[0]
#    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#    heatmap = tf.squeeze(heatmap)
#
#    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#    return heatmap.numpy()
#
#def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
#    img = cv2.imread(img_path)
#    img = cv2.resize(img, (img_width, img_height))
#    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#    heatmap = np.uint8(255 * heatmap)
#    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#    superimposed_img = heatmap_color * alpha + img
#    cv2.imwrite(cam_path, superimposed_img)
#    plt.imshow(cv2.cvtColor(superimposed_img.astype("uint8"), cv2.COLOR_BGR2RGB))
#    plt.axis("off")
#    plt.show()
#
# Grad-CAM テスト用
# img_path = "/Users/bitterchoco/fx/image-classification/test_image.jpg"
# img_array = get_img_array(img_path, size=(img_height, img_width))
# heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name=last_conv_layer_name)
# save_and_display_gradcam(img_path, heatmap)
