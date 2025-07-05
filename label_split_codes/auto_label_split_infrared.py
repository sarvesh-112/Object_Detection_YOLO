import os
from ultralytics import YOLO
import cv2

# Class mapping based on your use case
def map_class(name):
    name = name.lower()
    if "person" in name and "bike" in name:
        return 2  # person on bike
    elif "person" in name and "sit" in name:
        return 3  # person sitting
    elif "person" in name and ("car" in name or "vehicle" in name):
        return 1  # person in vehicle
    elif "person" in name:
        return 0  # person
    return None  # ignore other classes

def auto_label_images(img_dir, label_dir, model):
    os.makedirs(label_dir, exist_ok=True)
    images = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for file in images:
        img_path = os.path.join(img_dir, file)
        results = model(img_path)[0]

        label_path = os.path.join(label_dir, os.path.splitext(file)[0] + ".txt")
        with open(label_path, 'w') as f:
            for r in results.boxes.data:
                cls_id = int(r[5])
                name = model.names[cls_id]
                mapped_class = map_class(name)
                if mapped_class is None:
                    continue

                x_center, y_center, w, h = r[0:4]
                img = cv2.imread(img_path)
                h_img, w_img = img.shape[:2]

                # Normalize
                x_center = float(x_center) / w_img
                y_center = float(y_center) / h_img
                w = float(w) / w_img
                h = float(h) / h_img

                f.write(f"{mapped_class} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

if __name__ == "__main__":
    base_path = "C:/Users/smpga/PycharmProjects/Object_detection/datasets_split/infrared"
    model = YOLO("yolov8n.pt")  # or any other pretrained model

    print(" Auto-labeling train images...")
    auto_label_images(os.path.join(base_path, "images/train"), os.path.join(base_path, "labels/train"), model)

    print(" Auto-labeling val images...")
    auto_label_images(os.path.join(base_path, "images/val"), os.path.join(base_path, "labels/val"), model)

    print(" Infrared auto-labeling complete.")
