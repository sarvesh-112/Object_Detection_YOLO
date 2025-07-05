import os
import shutil
import random

def propagate_labels(img_dir, src_label_dir, dst_label_dir):
    os.makedirs(dst_label_dir, exist_ok=True)
    copied, missing = 0, []

    for file in os.listdir(img_dir):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        base_name = os.path.splitext(file)[0]
        label_file = base_name + ".txt"

        src_label_path = os.path.join(src_label_dir, label_file)
        dst_label_path = os.path.join(dst_label_dir, label_file)

        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dst_label_path)
            copied += 1
        else:
            missing.append(label_file)

    print(f" Copied {copied} label files.")
    if missing:
        print(f"️ {len(missing)} labels were missing. Sample:")
        print("\n".join(missing[:5]))

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
    enhanced_img_dir = "C:/Users/smpga/PycharmProjects/Object_detection/datasets/FUSED_Gradient_Enhanced_Advanced"
    visible_label_dir = "C:/Users/smpga/PycharmProjects/Object_detection/labels/visible"
    enhanced_label_dir = "C:/Users/smpga/PycharmProjects/Object_detection/labels/fused_enhanced"
    output_split_dir = "C:/Users/smpga/PycharmProjects/Object_detection/datasets_split/fused_gradient_enhanced_advanced"

    print(" Propagating labels...")
    propagate_labels(enhanced_img_dir, visible_label_dir, enhanced_label_dir)

    print("️ Splitting dataset into train/val...")
    split_dataset(enhanced_img_dir, enhanced_label_dir, output_split_dir, split_ratio=0.8)

    print(" Done: Dataset ready for YOLO training.")
