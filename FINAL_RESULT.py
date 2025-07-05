import pandas as pd
import os

# Define the final results data
data = {
    "Model": [
        "Visible_v8s",
        "Infrared_v8s",
        "Fused_Gradient_v8s",
        "Fused_MGF_v8s",
        "Fused_SF_v8s"
    ],
    "mAP50": [0.944, 0.256, 0.369, 0.836, 0.955],
    "mAP50-95": [0.854, 0.128, 0.268, 0.596, 0.785],
    "Precision": [0.862, 0.332, 0.309, 0.952, 0.894],
    "Recall": [0.909, 0.350, 0.382, 0.625, 0.847]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define the output directory and file path
output_dir = "final_results"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "final_results.csv")

# Save to CSV
df.to_csv(output_file, index=False)

print(f" Final results saved to: {output_file}")
