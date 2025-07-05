import os
import cv2
import numpy as np

def gamma_correction(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)
    ]).astype("uint8")
    return cv2.LUT(image, table)

def unsharp_mask(image, blur_ksize=(9, 9), strength=1.5):
    blurred = cv2.GaussianBlur(image, blur_ksize, 10.0)
    return cv2.addWeighted(image, strength, blurred, -0.5, 0)

def enhance_advanced(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Step 1: Gamma correction
        gamma_corrected = gamma_correction(img, gamma=1.5)

        # Step 2: Unsharp mask
        sharpened = unsharp_mask(gamma_corrected)

        # Save
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, sharpened)
        count += 1

    print(f" Enhanced {count} images using gamma + unsharp and saved to: {output_dir}")

if __name__ == "__main__":
    input_dir = "C:/Users/smpga/PycharmProjects/Object_detection/datasets/FUSED_Gradient"
    output_dir = "C:/Users/smpga/PycharmProjects/Object_detection/datasets/FUSED_Gradient_Enhanced_Advanced"

    enhance_advanced(input_dir, output_dir)
