import os
import pandas as pd

def extract_metrics_from_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"Missing file: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        metrics = df.iloc[-1]

        return {
            'mAP50': metrics.get('metrics/mAP50', metrics.get('map50', None)),
            'mAP50-95': metrics.get('metrics/mAP50-95', metrics.get('map', None)),
            'Precision': metrics.get('metrics/precision(B)', metrics.get('precision', None)),
            'Recall': metrics.get('metrics/recall(B)', metrics.get('recall', None)),
        }
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

if __name__ == "__main__":
    results = {
        "Visible": "runs/detect/visible_yolov8n/results.csv",
        "Infrared": "runs/detect/infrared_yolov8n/results.csv",
        "Fused_Gradient": "runs/detect/fused_gradient_yolov8n/results.csv",
        "Fused_Gradient_Enhanced": "runs/detect/fused_gradient_enhanced_yolov8n2/results.csv"
    }

    data = []
    for model_name, path in results.items():
        print(f"Checking: {model_name} → {path}")
        metrics = extract_metrics_from_csv(path)
        if metrics:
            row = {"Model": model_name}
            row.update(metrics)
            data.append(row)
        else:
            print(f"Skipping {model_name} — file not found or invalid.\n")

    if data:
        df = pd.DataFrame(data)
        print("\nYOLOv8 Model Comparison (Core Metrics Only):")
        print(df.to_string(index=False))
        df.to_csv("runs/comparison_metrics_core.csv", index=False)
        print("Saved to: runs/comparison_metrics_core.csv")
    else:
        print("No valid results found to compare.")
