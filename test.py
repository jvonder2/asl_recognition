import os
import random
import shutil

DEST = r"C:\Users\jvond\ML_Project\asl_alphabet_test"
SOURCE = r"C:\Users\jvond\ML_Project\asl_alphabet_train"
NUM_PER_CLASS = 250

for class_name in os.listdir(SOURCE):
    src_class_path = os.path.join(SOURCE, class_name)
    dst_class_path = os.path.join(DEST, class_name)

    if not os.path.isdir(src_class_path):
        continue

    os.makedirs(dst_class_path, exist_ok=True)

    images = [f for f in os.listdir(src_class_path)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if len(images) == 0:
        print(f"{class_name}: no images found, skipping")
        continue

    selected = random.sample(images, min(NUM_PER_CLASS, len(images)))

    for img in selected:
        src_file = os.path.join(src_class_path, img)
        dst_file = os.path.join(dst_class_path, img)
        shutil.copy2(src_file, dst_file)

    print(f"{class_name}: copied {len(selected)} images")

print("Done.")