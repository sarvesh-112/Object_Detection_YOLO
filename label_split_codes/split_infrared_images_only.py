import os
import shutil
import random

def split_images_only(image_dir, out_dir, split_ratio=0.8):
    images = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for phase, files in [("train", train_imgs), ("val", val_imgs)]:
        img_out = os.path.join(out_dir, f"images/{phase}")
        os.makedirs(img_out, exist_ok=True)

        for file in files:
            shutil.copy(os.path.join(image_dir, file), os.path.join(img_out, file))

if __name__ == "__main__":
    image_dir = "C:/Users/smpga/PycharmProjects/Object_detection/datasets/InfraRedImages"
    output_dir = "C:/Users/smpga/PycharmProjects/Object_detection/datasets_split/infrared"

    print(" Splitting infrared images into train/val sets (labels will be added later)...")
    split_images_only(image_dir, output_dir, split_ratio=0.8)
    print(" Split complete.")
