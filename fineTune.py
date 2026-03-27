"""
ASL Fine-Tuning Script
======================
Fine-tunes an existing ASL CNN model on a merged dataset that contains:
- my own hand images
- sampled Kaggle images

Expected folder structure:
    my_asl_dataset/
        A/
        B/
        C/
        ...
        Z/
        del/
        nothing/
        space/

Requirements:
    pip install tensorflow matplotlib scikit-learn
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.metrics import classification_report, confusion_matrix

# =========================================================
# CONFIG
# =========================================================
BASE_MODEL_PATH = "C:\\Users\\jvond\\ML_Project\\asl_model.keras"
DATA_DIR = "C:\\Users\\jvond\\ML_Project\\my_asl_dataset"
SAVE_DIR = "C:\\Users\\jvond\\ML_Project\\asl_finetuned"

FINETUNED_MODEL_PATH = f"{SAVE_DIR}/asl_model_finetuned.keras"
FINETUNE_HISTORY_PATH = f"{SAVE_DIR}/finetune_history.json"
CLASS_NAMES_PATH = f"{SAVE_DIR}/class_names_finetuned.json"

IMG_SIZE = 96
BATCH_SIZE = 16
EPOCHS = 8
VALIDATION_SPLIT = 0.2
SEED = 42

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================
print("Loading fine-tune dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

print(f"Found {NUM_CLASSES} classes:")
print(class_names)

with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(256).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# =========================================================
# LOAD BASE MODEL
# =========================================================
print("\nLoading base model...")
model = tf.keras.models.load_model(BASE_MODEL_PATH)
print("Loaded:", BASE_MODEL_PATH)

# =========================================================
# OPTIONAL: FREEZE SOME EARLY LAYERS
# =========================================================
# Since this is fine-tuning, freezing some early layers can help preserve
# basic learned features while adapting later layers to your hand/camera.
#
# If your model is a simple Sequential CNN, freezing the first few layers
# is a good starting point.

for layer in model.layers[:3]:
    layer.trainable = False

print("\nLayer trainable status:")
for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)

# =========================================================
# RECOMPILE WITH SMALL LEARNING RATE
# =========================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# =========================================================
# CALLBACKS
# =========================================================
my_callbacks = [
    callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    ),
    callbacks.ModelCheckpoint(
        FINETUNED_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]

# =========================================================
# FINE-TUNE
# =========================================================
print("\nStarting fine-tuning...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=my_callbacks,
)

# Load best saved model before evaluation
model = tf.keras.models.load_model(FINETUNED_MODEL_PATH)

# =========================================================
# SAVE TRAINING HISTORY
# =========================================================
hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(FINETUNE_HISTORY_PATH, "w") as f:
    json.dump(hist_dict, f, indent=2)

print(f"\nFine-tune history saved to {FINETUNE_HISTORY_PATH}")

# =========================================================
# EVALUATE ON VALIDATION SET
# =========================================================
print("\nEvaluating fine-tuned model on validation set...\n")

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(classification_report(y_true, y_pred, target_names=class_names))

# =========================================================
# PLOT TRAINING CURVES
# =========================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history["accuracy"], label="Train Accuracy")
ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
ax1.set_title("Fine-Tune Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history["loss"], label="Train Loss")
ax2.plot(history.history["val_loss"], label="Val Loss")
ax2.set_title("Fine-Tune Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/finetune_curves.png", dpi=150)
plt.show()

print("Fine-tune curves saved to finetune_curves.png")

# =========================================================
# CONFUSION MATRIX
# =========================================================
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.set_title("Fine-Tuned Model Confusion Matrix")

tick_marks = np.arange(NUM_CLASSES)
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(class_names, rotation=90, fontsize=8)
ax.set_yticklabels(class_names, fontsize=8)

ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")

plt.colorbar(im)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/finetune_confusion_matrix.png", dpi=150)
plt.show()

print("Fine-tune confusion matrix saved to finetune_confusion_matrix.png")

# =========================================================
# DONE
# =========================================================
print(f"""
Fine-tuning complete!

Files created:
  {FINETUNED_MODEL_PATH}         — Fine-tuned Keras model
  {FINETUNE_HISTORY_PATH}        — Fine-tuning history
  {CLASS_NAMES_PATH}             — Class label order
  finetune_curves.png            — Accuracy/loss plots
  finetune_confusion_matrix.png  — Confusion matrix

To use in inference:
  model = tf.keras.models.load_model("{FINETUNED_MODEL_PATH}")
""")