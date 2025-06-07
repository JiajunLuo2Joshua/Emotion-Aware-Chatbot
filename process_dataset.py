import os
import shutil
import random
from tqdm import tqdm

# configuration
SOURCE_AFFECTNET = "G:/1学习/dataset/affectnet-hq"
SOURCE_RAFDB = "G:/1学习/dataset/RAF_DB"
TARGET_DIR = "E:/COMPSYS731/dataset"

SPLIT_RATIO = 0.7  # 80% for training, 20% for validation
CATEGORIES = ['happy', 'sad', 'fear', 'anger', 'surprise', 'disgust', 'neutral']

# create the target directory
def prepare_target_dirs():
    for phase in ['train', 'val']:
        for category in CATEGORIES:
            os.makedirs(os.path.join(TARGET_DIR, phase, category), exist_ok=True)
    print(" Target directories created.")

# process a single source dataset
def copy_and_split(source_folder, source_name):
    """
    copy all the category files under source_folder to the target train/val directory
    """
    print(f"\n Processing dataset: {source_name}")

    for category in CATEGORIES:
        cat_path = os.path.join(source_folder, category)
        if not os.path.exists(cat_path):
            print(f" Warning: Category folder {cat_path} not found, skipping.")
            continue

        images = os.listdir(cat_path)
        if len(images) == 0:
            print(f" Warning: No images in {cat_path}, skipping...")
            continue

        random.shuffle(images)  # disrupt the order

        split_idx = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        print(f" {source_name}/{category}: {len(train_imgs)} train, {len(val_imgs)} val")

        # copy to train
        for img_name in tqdm(train_imgs, desc=f"{source_name} {category} - train"):
            src = os.path.join(cat_path, img_name)
            dst = os.path.join(TARGET_DIR, 'train', category, f"{source_name}_{img_name}")
            shutil.copy(src, dst)

        # copy to val
        for img_name in tqdm(val_imgs, desc=f"{source_name} {category} - val"):
            src = os.path.join(cat_path, img_name)
            dst = os.path.join(TARGET_DIR, 'val', category, f"{source_name}_{img_name}")
            shutil.copy(src, dst)

if __name__ == "__main__":
    random.seed(42)  # ensure that the division is consistent in each operation

    print(" Preparing target folders...")
    prepare_target_dirs()

    copy_and_split(SOURCE_AFFECTNET, "affectnet")
    copy_and_split(SOURCE_RAFDB, "rafdb")

    print("\n Dataset processing complete!")
