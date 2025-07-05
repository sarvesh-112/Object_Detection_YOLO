import cv2
import os
import numpy as np

def fuse_images_laplacian(img1, img2):
    lap1 = cv2.Laplacian(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
    lap2 = cv2.Laplacian(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
    mask = np.abs(lap1) >= np.abs(lap2)
    fused = img1.copy()
    fused[~mask] = img2[~mask]
    return fused

def run_laplacian_fusion(vis_dir, ir_dir, output_dir):
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
        fused = fuse_images_laplacian(vis_img, ir_img)
        cv2.imwrite(os.path.join(output_dir, file), fused)

if __name__ == "__main__":
    run_laplacian_fusion(
        vis_dir="C:/Users/smpga/PycharmProjects/Object_detection/datasets/VisibleImages",
        ir_dir="C:/Users/smpga/PycharmProjects/Object_detection/datasets/InfraRedImages",
        output_dir="C:/Users/smpga/PycharmProjects/Object_detection/datasets/FUSED_Laplacian"
    )
