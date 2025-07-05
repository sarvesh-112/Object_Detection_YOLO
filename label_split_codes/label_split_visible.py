import os
import shutil
import random

def split_dataset(image_dir, label_dir, out_dir, split_ratio=0.8):
    images = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for phase, files in [("train", train_imgs), ("val", val_imgs)]:
        img_out = os.path.join(out_dir, f"images/{phase}")
        label_out = os.path.join(out_dir, f"labels/{phase}")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(label_out, exist_ok=True)

        for file in files:
            shutil.copy(os.path.join(image_dir, file), os.path.join(img_out, file))

            label_file = os.path.splitext(file)[0] + ".txt"
            label_src = os.path.join(label_dir, label_file)
            label_dst = os.path.join(label_out, label_file)

            if os.path.exists(label_src):
                shutil.copy(label_src, label_dst)
            else:
                print(f"️ Label not found for image: {file}")

if __name__ == "__main__":
    image_dir = "C:/Users/smpga/PycharmProjects/Object_detection/datasets/VisibleImages"
    label_dir = "C:/Users/smpga/PycharmProjects/Object_detection/labels/visible"
    output_dir = "C:/Users/smpga/PycharmProjects/Object_detection/datasets_split/visible"

    print(" Splitting Visible dataset into train/val...")
    split_dataset(image_dir, label_dir, output_dir, split_ratio=0.8)
    print(" Visible dataset split complete.")
