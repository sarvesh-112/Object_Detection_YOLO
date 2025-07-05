import os
import cv2

def enhance_fused_images_clahe(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    count = 0

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        cv2.imwrite(os.path.join(output_dir, filename), enhanced_img)
        count += 1

    print(f" Enhanced {count} images and saved to {output_dir}")

if __name__ == "__main__":
    input_dir = "C:/Users/smpga/PycharmProjects/Object_detection/datasets/FUSED_Gradient"
    output_dir = "C:/Users/smpga/PycharmProjects/Object_detection/datasets/FUSED_Gradient_Enhanced"
    enhance_fused_images_clahe(input_dir, output_dir)
