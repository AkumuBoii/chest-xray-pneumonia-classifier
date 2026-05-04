import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Point to data folder
DATA_DIR = r"C:\Users\acer\OneDrive\Desktop\xray project\chest_xray\train"
NORMAL_DIR = r"C:\Users\acer\OneDrive\Desktop\xray project\chest_xray\train\NORMAL"
PNEUMONIA_DIR = r"C:\Users\acer\OneDrive\Desktop\xray project\chest_xray\train\PNEUMONIA"

# Count images
print(f"Normal X-rays:    {len(os.listdir(NORMAL_DIR))}")
print(f"Pneumonia X-rays: {len(os.listdir(PNEUMONIA_DIR))}")

# Display a few examples side by side
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle("Chest X-ray Examples", fontsize=14)

for i, fname in enumerate(os.listdir(NORMAL_DIR)[:3]):
    img = Image.open(os.path.join(NORMAL_DIR, fname)).convert("L")
    axes[0, i].imshow(img, cmap="gray")
    axes[0, i].set_title("Normal")
    axes[0, i].axis("off")

for i, fname in enumerate(os.listdir(PNEUMONIA_DIR)[:3]):
    img = Image.open(os.path.join(PNEUMONIA_DIR, fname)).convert("L")
    axes[1, i].imshow(img, cmap="gray")
    axes[1, i].set_title("Pneumonia")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data loaders with augmentation
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True,
    zoom_range=0.1
)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    r"C:\Users\acer\OneDrive\Desktop\xray project\chest_xray\train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data = test_gen.flow_from_directory(
    r"C:\Users\acer\OneDrive\Desktop\xray project\chest_xray\val",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# Load pretrained MobileNetV2
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# Build classifier on top
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train the model (takes 10-30 min)
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

# Plot training progress
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy over epochs")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss over epochs")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate on test set
from sklearn.metrics import classification_report
import numpy as np

test_data = test_gen.flow_from_directory(
    r"C:\Users\acer\OneDrive\Desktop\xray project\chest_xray\test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

predictions = (model.predict(test_data) > 0.5).astype(int)
true_labels = test_data.classes

print(classification_report(true_labels, predictions,
      target_names=["Normal", "Pneumonia"]))

# Save the model
model.save("xray_classifier.h5")

