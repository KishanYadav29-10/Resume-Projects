import os, shutil
from sklearn.model_selection import train_test_split

source_dir = "dataset/train"
val_dir = "dataset/val"

os.makedirs(val_dir, exist_ok=True)

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=0)

    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    for img in val_imgs:
        src = os.path.join(class_path, img)
        dst = os.path.join(val_dir, class_name, img)
        shutil.copy2(src, dst)

print("Validation data successfully split and copied.")
