import cv2
import numpy as np
import os

def guided_filter(I, p, r, eps):
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * I + mean_b
    return q

def mgf_fusion(vis_img, ir_img, r=15, eps=1e-3):
    vis_gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY) / 255.0
    ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY) / 255.0

    base_vis = guided_filter(vis_gray, vis_gray, r, eps)
    base_ir = guided_filter(ir_gray, ir_gray, r, eps)

    detail_vis = vis_gray - base_vis
    detail_ir = ir_gray - base_ir

    fused_base = np.maximum(base_vis, base_ir)
    fused_detail = np.maximum(detail_vis, detail_ir)

    fused_gray = np.clip(fused_base + fused_detail, 0, 1)

    fused_ycrcb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2YCrCb)
    fused_ycrcb[..., 0] = (fused_gray * 255).astype(np.uint8)
    fused_bgr = cv2.cvtColor(fused_ycrcb, cv2.COLOR_YCrCb2BGR)

    return fused_bgr

def run_mgf_fusion(vis_dir, ir_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(os.listdir(vis_dir))
    fused_count = 0

    for file in files:
        if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        vis_path = os.path.join(vis_dir, file)
        ir_path = os.path.join(ir_dir, file)
        if not os.path.exists(ir_path):
            continue

        vis_img = cv2.imread(vis_path)
        ir_img = cv2.imread(ir_path)
        if vis_img is None or ir_img is None:
            continue

        vis_img = cv2.resize(vis_img, (ir_img.shape[1], ir_img.shape[0]))
        fused = mgf_fusion(vis_img, ir_img)

        cv2.imwrite(os.path.join(output_dir, file), fused)
        fused_count += 1

    print(f" Fused {fused_count} image pairs with MGF.")
    print(f" Saved to: {output_dir}")

if __name__ == "__main__":
    vis_dir = "C:/Users/smpga/PycharmProjects/Object_detection/datasets/VisibleImages"
    ir_dir = "C:/Users/smpga/PycharmProjects/Object_detection/datasets/InfraRedImages"
    output_dir = "C:/Users/smpga/PycharmProjects/Object_detection/datasets/FUSED_MGF"
    run_mgf_fusion(vis_dir, ir_dir, output_dir)
