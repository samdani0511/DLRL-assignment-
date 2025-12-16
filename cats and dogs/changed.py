# ============================
# 1. IMPORTS
# ============================
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import kagglehub

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================
# 2. DOWNLOAD DATASET (KAGGLEHUB)
# ============================
print("Downloading dataset using KaggleHub...")
dataset_path = kagglehub.dataset_download("birajsth/cats-and-dogs-filtered")
print("Dataset downloaded at:", dataset_path)

base_dir = os.path.join(dataset_path, "cats_and_dogs_filtered")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

assert os.path.exists(train_dir), "Train directory not found!"
assert os.path.exists(val_dir), "Validation directory not found!"

# ============================
# 3. DATA GENERATORS
# ============================
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print("Class indices:", train_generator.class_indices)

# ============================
# 4. CNN MODEL
# ============================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(150,150,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.summary()

# ============================
# 5. COMPILE MODEL
# ============================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ============================
# 6. CALLBACKS
# ============================
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(patience=3, factor=0.3)

# ============================
# 7. TRAIN MODEL
# ============================
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=[early_stop, reduce_lr]
)

# ============================
# 8. SINGLE IMAGE PREDICTION
# ============================
random_class = random.choice(['cats', 'dogs'])
img_dir = os.path.join(train_dir, random_class)
img_file = random.choice(os.listdir(img_dir))

img_path = os.path.join(img_dir, img_file)

img = load_img(img_path, target_size=IMG_SIZE)
x = img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

pred = model.predict(x)[0][0]

print("Actual class:", random_class)
print("Predicted:", "Dog" if pred > 0.5 else "Cat")

# ============================
# 9. TRAINING CURVES
# ============================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure()
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title("Training vs Validation Accuracy")
plt.legend(["Train", "Validation"])

plt.figure()
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title("Training vs Validation Loss")
plt.legend(["Train", "Validation"])
plt.show()
