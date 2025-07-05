import os
import cv2

# Paths
fused_image_dir = "datasets/FUSED_Gradient"
fused_label_dir = "labels/fused"

def draw_yolo_boxes(img_path, label_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f" Failed to load: {img_path}")
        return

    h, w = img.shape[:2]
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = map(float, parts)
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, str(int(cls)), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("YOLO Label Check", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_check():
    sample_files = sorted([f for f in os.listdir(fused_image_dir) if f.lower().endswith((".jpg", ".png"))])
    print(f"Checking {len(sample_files)} images... Press any key to go next.")

    for file in sample_files[:10]:  # Check first 10 samples
        img_path = os.path.join(fused_image_dir, file)
        label_path = os.path.join(fused_label_dir, os.path.splitext(file)[0] + ".txt")
        if os.path.exists(label_path):
            draw_yolo_boxes(img_path, label_path)
        else:
            print(f"⚠️ No label found for: {file}")

if __name__ == "__main__":
    run_check()
