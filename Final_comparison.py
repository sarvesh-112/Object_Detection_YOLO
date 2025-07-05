import os
import pandas as pd

# Define model result paths
model_dirs = {
    "Visible_v8s": "runs_advanced/visible_yolov8s/results.csv",
    "Infrared_v8s": "runs_advanced/infrared_yolov8s/results.csv",
    "Fused_SF_v8s": "runs_advanced/fused_sf_yolov8s2/results.csv",
    "Fused_MGF_v8s": "runs_advanced/fused_mgf_yolov8s/results.csv",
    "Fused_Gradient_Advanced_v8s": "runs_advanced/fused_gradient_enhanced_advanced_yolov8s/results.csv"
}

# Gather metrics into a list
results = []

for label, path in model_dirs.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        last_row = df.iloc[-1]  # Use final epoch metrics

        results.append({
            "Model": label,
            "mAP50": round(last_row.get("metrics/mAP50(B)", float('nan')), 4),
            "mAP50-95": round(last_row.get("metrics/mAP50-95(B)", float('nan')), 4),
            "Precision": round(last_row.get("metrics/precision(B)", float('nan')), 4),
            "Recall": round(last_row.get("metrics/recall(B)", float('nan')), 4)
        })
    else:
        print(f"File not found: {path}")
        results.append({
            "Model": label,
            "mAP50": None,
            "mAP50-95": None,
            "Precision": None,
            "Recall": None
        })

# Convert to DataFrame and save
comparison_df = pd.DataFrame(results)
print("\nYOLOv8 Model Comparison:\n")
print(comparison_df.to_string(index=False))

# Save to CSV
output_path = "runs_advanced/final_comparison.csv"
comparison_df.to_csv(output_path, index=False)
print(f"\nSaved comparison to: {output_path}")
