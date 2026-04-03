"""
train.py — Train the Plant Disease Detection Model
----------------------------------------------------
WHAT THIS FILE DOES:
  1. Loads and augments the PlantVillage dataset
  2. Builds a CNN using Transfer Learning (MobileNetV2)
  3. Trains the model in two phases:
       Phase 1 — Train only the top layers (fast)
       Phase 2 — Fine-tune the base model (more accurate)
  4. Saves the best model to models/plant_disease_model.h5

HOW TO RUN:
  cd src
  python train.py

EXPECTED OUTPUT:
  ~90%+ validation accuracy after 15 epochs
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import plot_history, CLASS_NAMES

# ── 1. CONFIGURATION ──────────────────────────────────────────────────────────
# These are hyperparameters — values you can tune to improve results.

IMG_SIZE    = 224        # MobileNetV2 expects 224x224 images
BATCH_SIZE  = 32         # Number of images processed at once (reduce to 16 if RAM issues)
EPOCHS_P1   = 10         # Phase 1 epochs (top layers only)
EPOCHS_P2   = 5          # Phase 2 epochs (fine-tuning)
NUM_CLASSES = 38         # PlantVillage has 38 disease/healthy categories
DATA_DIR    = "../dataset"
MODEL_PATH  = "../models/plant_disease_model.h5"

os.makedirs("../models", exist_ok=True)


# ── 2. DATA AUGMENTATION & LOADING ────────────────────────────────────────────
# WHY AUGMENTATION?
#   Our model should recognize diseases even if the leaf image is:
#   - rotated (farmer holds phone at angle)
#   - flipped (left/right doesn't change the disease)
#   - zoomed in (closeup vs far shot)
#   Augmentation artificially creates these variations during training.

print("\n📂 Loading dataset...")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,        # Normalize pixel values from [0,255] → [0,1]
    rotation_range=30,         # Randomly rotate images up to 30 degrees
    width_shift_range=0.2,     # Randomly shift images horizontally
    height_shift_range=0.2,    # Randomly shift images vertically
    shear_range=0.2,           # Shear transformation (slant the image)
    zoom_range=0.2,            # Random zoom in/out
    horizontal_flip=True,      # Randomly flip left-right
    fill_mode="nearest",       # Fill empty pixels after transformation
    validation_split=0.2,      # 80% train, 20% validation
)

# Validation data: only rescale, NO augmentation
# WHY? We want to evaluate on real, unmodified images
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",   # One-hot encoded labels for multi-class
    subset="training",
    shuffle=True,
)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

print(f"✅ Training samples   : {train_generator.samples}")
print(f"✅ Validation samples : {val_generator.samples}")
print(f"✅ Classes found      : {len(train_generator.class_indices)}")


# ── 3. BUILD THE MODEL ────────────────────────────────────────────────────────
# WHY TRANSFER LEARNING?
#   Training a CNN from scratch needs millions of images and hours of GPU time.
#   MobileNetV2 is already trained on 1.4 million ImageNet images and has
#   learned to detect edges, textures, shapes, and objects.
#   We "transfer" this knowledge to our plant disease task.
#
# ARCHITECTURE:
#   Input (224x224x3)
#       ↓
#   MobileNetV2 base (frozen) ← pretrained feature extractor
#       ↓
#   GlobalAveragePooling2D    ← compress feature maps to a vector
#       ↓
#   Dense(256, ReLU)          ← learn disease-specific patterns
#       ↓
#   Dropout(0.3)              ← prevent overfitting
#       ↓
#   Dense(38, Softmax)        ← output probabilities for 38 classes

print("\n🧠 Building model...")

# Load MobileNetV2 without its top classification layer
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,      # Remove original 1000-class ImageNet head
    weights="imagenet",     # Use pretrained weights
)
base_model.trainable = False  # Freeze: don't update these weights yet

# Add our custom classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),        # (7, 7, 1280) → (1280,)
    layers.Dense(256, activation="relu"),   # Learn plant-specific features
    layers.Dropout(0.3),                    # Randomly drop 30% neurons (prevents overfitting)
    layers.Dense(NUM_CLASSES, activation="softmax"),  # 38 probability outputs
], name="PlantDiseaseDetector")

model.summary()


# ── 4. PHASE 1 TRAINING — Top Layers Only ────────────────────────────────────
# WHY TWO PHASES?
#   Phase 1: Freeze the base model, train only the new head.
#            This is fast and prevents destroying the pretrained weights.
#   Phase 2: Unfreeze some base layers and fine-tune with a tiny learning rate.
#            This squeezes out extra accuracy.

print(f"\n🚀 Phase 1: Training top layers for {EPOCHS_P1} epochs...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    # Stop early if validation loss stops improving (saves time)
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    ),
    # Save the best model automatically
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    # Reduce learning rate if stuck
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1
    ),
]

history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_P1,
    callbacks=callbacks,
)


# ── 5. PHASE 2 TRAINING — Fine-Tuning ────────────────────────────────────────
print(f"\n🔬 Phase 2: Fine-tuning last 30 layers for {EPOCHS_P2} epochs...")

# Unfreeze the last 30 layers of MobileNetV2
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Use a much smaller learning rate to avoid destroying pretrained weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_P2,
    callbacks=callbacks,
)


# ── 6. EVALUATE & SAVE ───────────────────────────────────────────────────────
print("\n📊 Evaluating on validation set...")
val_loss, val_acc = model.evaluate(val_generator)
print(f"\n✅ Final Validation Accuracy : {val_acc * 100:.2f}%")
print(f"✅ Final Validation Loss     : {val_loss:.4f}")
print(f"\n💾 Model saved → {MODEL_PATH}")

# Plot training curves
plot_history(history1)
