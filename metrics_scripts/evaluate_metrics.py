import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

def yang_metric(img1, img2, fused):
    return np.mean(np.abs(fused - img1)) + np.mean(np.abs(fused - img2))

def compute_entropy(image):
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256), density=True)
    return entropy(hist + 1e-10)

def mutual_information(img1, img2):
    hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=256)
    return mutual_info_score(None, None, contingency=hist_2d)

def qabf(fused, img1, img2):
    grad_fused = np.gradient(fused.astype(np.float32))
    grad_img1 = np.gradient(img1.astype(np.float32))
    grad_img2 = np.gradient(img2.astype(np.float32))
    q1 = np.sum(np.minimum(grad_fused[0], grad_img1[0]) + np.minimum(grad_fused[1], grad_img1[1]))
    q2 = np.sum(np.minimum(grad_fused[0], grad_img2[0]) + np.minimum(grad_fused[1], grad_img2[1]))
    return (q1 + q2) / (np.sum(grad_fused[0] + grad_fused[1]) + 1e-10)

def evaluate_metrics_all(fused_dir, vis_dir, ir_dir):
    results = []
    for file in sorted(os.listdir(fused_dir)):
        if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        fused = cv2.imread(os.path.join(fused_dir, file), cv2.IMREAD_GRAYSCALE)
        vis = cv2.imread(os.path.join(vis_dir, file), cv2.IMREAD_GRAYSCALE)
        ir = cv2.imread(os.path.join(ir_dir, file), cv2.IMREAD_GRAYSCALE)
        if fused is None or vis is None or ir is None:
            continue
        fused = cv2.resize(fused, (vis.shape[1], vis.shape[0]))
        results.append({
            'Image': file,
            'Yang': yang_metric(vis, ir, fused),
            'SSIM': ssim(vis, fused),
            'Entropy': compute_entropy(fused),
            'MI': mutual_information(vis, fused),
            'Qabf': qabf(fused, vis, ir)
        })
    df = pd.DataFrame(results)
    df.loc['Average'] = df.mean(numeric_only=True)
    return df

# Paths
base = "C:/Users/smpga/PycharmProjects/Object_detection/datasets"
visible_dir = os.path.join(base, "VisibleImages")
infrared_dir = os.path.join(base, "InfraRedImages")

fused_dirs = {
    "Fused_SF": os.path.join(base, "FUSED_SF"),
    "Fused_Laplacian": os.path.join(base, "FUSED_Laplacian"),
    "Fused_Gradient": os.path.join(base, "FUSED_Gradient"),
    "Fused_MGF": os.path.join(base, "FUSED_MGF"),
}

# Compare all
all_results = {}
for name, path in fused_dirs.items():
    print(f"Evaluating metrics for: {name}")
    df = evaluate_metrics_all(path, visible_dir, infrared_dir)
    all_results[name] = df.loc['Average']

# Save final comparison
final_df = pd.DataFrame(all_results).T
final_df.to_csv("fusion_metrics_comparison.csv")
print("\nSaved to fusion_metrics_comparison.csv")
print(final_df)
