import os
import shutil
import random

# ===============================
# CONFIG — CHANGE THESE
# ===============================
KAGGLE_DIR = "C:\\Users\\jvond\\ML_Project\\asl_alphabet_train"
TARGET_DIR = "C:\\Users\\jvond\\ML_Project\\my_asl_dataset"

SAMPLES_PER_CLASS = 250
SEED = 42

random.seed(SEED)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ===============================
# HELPERS
# ===============================
def list_images(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in VALID_EXTS
    ]

# ===============================
# MAIN
# ===============================
classes = sorted([
    d for d in os.listdir(TARGET_DIR)
    if os.path.isdir(os.path.join(TARGET_DIR, d))
])

print("Classes found:")
print(classes)

for cls in classes:
    print(f"\nProcessing class: {cls}")

    kaggle_cls = os.path.join(KAGGLE_DIR, cls)
    target_cls = os.path.join(TARGET_DIR, cls)

    if not os.path.exists(kaggle_cls):
        print(f"  ⚠️ Skipping {cls} (not found in Kaggle)")
        continue

    kaggle_files = list_images(kaggle_cls)
    print(f"  Kaggle total: {len(kaggle_files)}")

    if len(kaggle_files) > SAMPLES_PER_CLASS:
        kaggle_files = random.sample(kaggle_files, SAMPLES_PER_CLASS)

    print(f"  Copying {len(kaggle_files)} images...")

    existing_files = set(os.listdir(target_cls))

    count = 0
    for i, src in enumerate(kaggle_files):
        ext = os.path.splitext(src)[1]
        dst_name = f"kaggle_{i}_{random.randint(0,99999)}{ext}"
        dst_path = os.path.join(target_cls, dst_name)

        shutil.copy2(src, dst_path)
        count += 1

    print(f"  Added {count} images to {cls}")

print("\n✅ Done merging Kaggle images into your dataset!")