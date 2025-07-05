import os
import pandas as pd

def load_metrics(csv_path):
    try:
        df = pd.read_csv(csv_path)
        metrics = df.iloc[0]
        return {
            "mAP50": round(metrics.get("metrics/mAP50(B)", 0), 4),
            "mAP50-95": round(metrics.get("metrics/mAP50-95(B)", 0), 4),
            "Precision": round(metrics.get("metrics/precision(B)", 0), 4),
            "Recall": round(metrics.get("metrics/recall(B)", 0), 4)
        }
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return {"mAP50": None, "mAP50-95": None, "Precision": None, "Recall": None}

def main():
    base_dir = "runs_advanced"
    models = {
        "Visible_v8s": os.path.join(base_dir, "visible_yolov8s", "results.csv"),
        "Infrared_v8s": os.path.join(base_dir, "infrared_yolov8s", "results.csv"),
        "Fused_Gradient_v8s": os.path.join(base_dir, "fused_gradient_yolov8s", "results.csv"),
        "Advanced_Fused_v8s": os.path.join(base_dir, "fused_gradient_enhanced_advanced_yolov8s", "results.csv")
    }

    comparison_data = []
    for name, path in models.items():
        metrics = load_metrics(path)
        comparison_data.append({
            "Model": name,
            **metrics
        })

    df = pd.DataFrame(comparison_data)
    print("\nYOLOv8s Model Comparison (Advanced):\n")
    print(df.to_string(index=False))

    out_path = os.path.join(base_dir, "comparison_advanced.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved comparison to: {out_path}")

if __name__ == "__main__":
    main()
