import os
import cv2
from ultralytics import YOLO

# === Paths ===
image_dir = "C:/Users/smpga/PycharmProjects/Object_detection/datasets/VisibleImages"
output_label_dir = "C:/Users/smpga/PycharmProjects/Object_detection/labels/visible"
os.makedirs(output_label_dir, exist_ok=True)

# === Load YOLOv8 model ===
model = YOLO("yolov8n.pt")  # or yolov8s.pt for better results

def compute_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0

def process_image(file):
    image_path = os.path.join(image_dir, file)
    img = cv2.imread(image_path)
    if img is None:
        return

    height, width = img.shape[:2]
    results = model(image_path)[0]

    persons, cars, bikes = [], [], []
    labels = []

    for box in results.boxes:
        cls_id = int(box.cls)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf)
        if conf < 0.4:
            continue

        if cls_id == 0: persons.append([x1, y1, x2, y2])
        elif cls_id == 2: cars.append([x1, y1, x2, y2])
        elif cls_id in [3, 4]: bikes.append([x1, y1, x2, y2])  # motorcycle, bicycle

    for p_box in persons:
        px1, py1, px2, py2 = p_box
        pw, ph = px2 - px1, py2 - py1
        cx = (px1 + px2) / 2 / width
        cy = (py1 + py2) / 2 / height
        nw = pw / width
        nh = ph / height

        # Default class
        class_id = 0  # person (standing)

        if any(compute_iou(p_box, c) > 0.3 for c in cars):
            class_id = 1  # person in vehicle
        elif any(compute_iou(p_box, b) > 0.3 for b in bikes):
            class_id = 2  # person on bike
        elif ph / pw < 1.2:
            class_id = 3  # person sitting (aspect heuristic)

        labels.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    # Write YOLO label file
    label_file = os.path.splitext(file)[0] + ".txt"
    with open(os.path.join(output_label_dir, label_file), "w") as f:
        f.write("\n".join(labels))

    print(f" {file} → {len(labels)} objects")

# === Run for all visible images ===
if __name__ == "__main__":
    files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))])
    for file in files:
        process_image(file)
    print("\n Auto-labeling complete. Labels saved to: labels/visible/")
