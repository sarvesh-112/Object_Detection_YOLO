import cv2
import os
import numpy as np

def sobel_energy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(sobelx**2 + sobely**2)

def fuse_images_gradient(img1, img2):
    energy1 = sobel_energy(img1)
    energy2 = sobel_energy(img2)
    mask = energy1 >= energy2
    fused = img1.copy()
    fused[~mask] = img2[~mask]
    return fused

def run_gradient_fusion(vis_dir, ir_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(os.listdir(vis_dir))

    for file in files:
        vis_path = os.path.join(vis_dir, file)
        ir_path = os.path.join(ir_dir, file)
        if not os.path.exists(ir_path): continue

        vis_img = cv2.imread(vis_path)
        ir_img = cv2.imread(ir_path)
        if vis_img is None or ir_img is None: continue

        vis_img = cv2.resize(vis_img, (ir_img.shape[1], ir_img.shape[0]))
        fused = fuse_images_gradient(vis_img, ir_img)
        cv2.imwrite(os.path.join(output_dir, file), fused)

if __name__ == "__main__":
    run_gradient_fusion(
        vis_dir="C:/Users/smpga/PycharmProjects/Object_detection/datasets/VisibleImages",
        ir_dir="C:/Users/smpga/PycharmProjects/Object_detection/datasets/InfraRedImages",
        output_dir="C:/Users/smpga/PycharmProjects/Object_detection/datasets/FUSED_Gradient"
    )
