"""
ASL Alphabet CNN Classifier — Training Script
================================================
This script trains a Convolutional Neural Network to classify 29 ASL hand signs.

Usage:
    1. Download the dataset from https://www.kaggle.com/datasets/grassknoted/asl-alphabet
    2. Unzip it so you have a folder structure like:
         asl_alphabet_train/
            A/
            B/
            ...
            space/
            del/
            nothing/
    3. Update DATA_DIR below to point to your training folder.
    4. Run:  python asl_cnn_train.py

Requirements:
    pip install tensorflow matplotlib scikit-learn
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR = "asl_alphabet_train"  # path to the training images folder
IMG_SIZE = 96          # match the dataset's native 96x96 resolution
BATCH_SIZE = 8          # lower batch size can help with overfitting, but will take longer to train
EPOCHS = 20             # 15-20 is usually enough; early stopping will halt if it plateaus
VALIDATION_SPLIT = 0.2  # 80% train, 20% validation
MODEL_SAVE_PATH = "asl_model.keras"          # full model (architecture + weights)
TFLITE_SAVE_PATH = "asl_model.tflite"        # lightweight version for deployment
HISTORY_SAVE_PATH = "training_history.json"   # training metrics for your paper


# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS DATA
# ─────────────────────────────────────────────
print("Loading dataset...")

# Training set (80%)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
)

# Validation set (20%)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
)

# Save class names so you can map predictions → letters later
class_names = train_ds.class_names
print(f"Found {len(class_names)} classes: {class_names}")
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# Performance optimization: prefetch data so GPU never waits on CPU
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

NUM_CLASSES = len(class_names)


# ─────────────────────────────────────────────
# 2. DATA AUGMENTATION (helps prevent overfitting)
# ─────────────────────────────────────────────
# This is especially important for your project since the Kaggle dataset
# has uniform backgrounds — augmentation simulates real-world variation.
# Took this out it was causing too much overhead with RAM usage and the dataset wasn't learning with augmentation
data_augmentation = tf.keras.Sequential([
    #layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),          # up to ~27° rotation
    layers.RandomZoom(0.1),               # slight zoom in/out
    layers.RandomBrightness(0.2),         # brightness variation
    layers.RandomContrast(0.2),           # contrast variation
    layers.RandomTranslation(0.1, 0.1),   # slight shifts
])


# ─────────────────────────────────────────────
# 3. BUILD THE CNN MODEL
# ─────────────────────────────────────────────
def build_model():
    """
    Architecture:
        - Rescaling (normalize pixels 0-255 → 0-1)
        - Data augmentation
        - 4 Convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPool)
          with increasing filter counts: 32 → 64 → 128 → 256
        - Dropout for regularization
        - Dense classifier head with 29 outputs

    This is a step up from the basic 3-layer architecture in your doc.
    The extra block + BatchNorm + Dropout will help a lot with the
    overfitting concern your doc mentions.
    """
    model = models.Sequential([
        # --- Input & Preprocessing ---
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Rescaling(1.0 / 255),   # normalize pixel values

        # --- Block 1: 32 filters ---
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # --- Block 2: 64 filters ---
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # --- Block 3: 128 filters ---
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # --- Block 4: 256 filters ---
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # --- Classifier Head ---
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    return model

import tensorflow as tf

model = tf.keras.models.load_model("asl_model.keras")
model.summary()


# ─────────────────────────────────────────────
# 4. COMPILE THE MODEL
# ─────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# ─────────────────────────────────────────────
# 5. SET UP CALLBACKS
# ─────────────────────────────────────────────
my_callbacks = [
    # Stop training early if validation accuracy stops improving
    callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,            # wait 5 epochs before giving up
        restore_best_weights=True,
    ),
    # Reduce learning rate when validation loss plateaus
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,            # halve the learning rate
        patience=3,
        min_lr=1e-6,
    ),
    # Save the best model during training
    callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]


# ─────────────────────────────────────────────
# 6. TRAIN THE MODEL
# ─────────────────────────────────────────────
print("\n Starting training...\n")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=my_callbacks,
)

# Save training history for your paper's graphs
hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(HISTORY_SAVE_PATH, "w") as f:
    json.dump(hist_dict, f, indent=2)
print(f"\nTraining history saved to {HISTORY_SAVE_PATH}")


# ─────────────────────────────────────────────
# 7. EVALUATE ON VALIDATION SET
# ─────────────────────────────────────────────
print("\n Evaluating model on validation set...\n")

# Gather all true labels and predictions
y_true = []
y_pred = []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Print classification report (precision, recall, F1 per class)
print(classification_report(y_true, y_pred, target_names=class_names))


# ─────────────────────────────────────────────
# 8. PLOT TRAINING CURVES (for your paper)
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
ax1.plot(history.history["accuracy"], label="Train Accuracy")
ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
ax1.set_title("Model Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(history.history["loss"], label="Train Loss")
ax2.plot(history.history["val_loss"], label="Val Loss")
ax2.set_title("Model Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("Training curves saved to training_curves.png")


# ─────────────────────────────────────────────
# 9. CONFUSION MATRIX (for your paper)
# ─────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.set_title("Confusion Matrix")
tick_marks = np.arange(NUM_CLASSES)
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(class_names, rotation=90, fontsize=8)
ax.set_yticklabels(class_names, fontsize=8)
ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")
plt.colorbar(im)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix saved to confusion_matrix.png")


# ─────────────────────────────────────────────
# 10. EXPORT TO TFLITE (optional, for faster inference)
# ─────────────────────────────────────────────
print("\nConverting to TFLite for faster inference...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(TFLITE_SAVE_PATH, "wb") as f:
    f.write(tflite_model)
print(f"TFLite model saved to {TFLITE_SAVE_PATH} ({len(tflite_model) / 1e6:.1f} MB)")


# ─────────────────────────────────────────────
# DONE!
# ─────────────────────────────────────────────
print(f""" Training complete!

Files created:
  {MODEL_SAVE_PATH}       — Full Keras model (use this for loading in your project)
  {TFLITE_SAVE_PATH}      — Lightweight model (optional, for faster inference)
  class_names.json         — List of class labels in order
  {HISTORY_SAVE_PATH}  — Raw metrics for graphing in your paper
  training_curves.png      — Accuracy & loss plots
  confusion_matrix.png     — Per-class confusion matrix

To load the model later in another script:
  import tensorflow as tf
  model = tf.keras.models.load_model("{MODEL_SAVE_PATH}")

To predict on a single image:
  import numpy as np
  img = tf.keras.utils.load_img("hand.jpg", target_size=({IMG_SIZE}, {IMG_SIZE}))
  img_array = tf.keras.utils.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
  predictions = model.predict(img_array)
  predicted_class = class_names[np.argmax(predictions)]
  confidence = np.max(predictions)
  print(f"Predicted: {{predicted_class}} ({{confidence:.1%}})")
""")