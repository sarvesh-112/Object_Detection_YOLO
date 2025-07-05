import pandas as pd

# Path to metrics_results.csv
csv_path = "C:/Users/smpga/PycharmProjects/Object_detection/metrics_scripts/metrics_results.csv"

# Load the results
df = pd.read_csv(csv_path)

# Metrics to average
metrics = ["Yang", "Entropy", "Qabf"]

# Compute average metrics per method
avg_df = df.groupby("Method")[metrics].mean().reset_index()
avg_df = avg_df.round(4)  # Optional: round to 4 decimal places

print(" Average Metric Values by Fusion Method:\n")
print(avg_df)

# Find best method per metric
print("\n Best Method per Metric:")
for metric in metrics:
    best_row = avg_df.loc[avg_df[metric].idxmax()]
    print(f"  - {metric}: {best_row['Method']} ({best_row[metric]})")

# Optional: Overall score (normalized and summed)
norm_df = avg_df.copy()
for metric in metrics:
    norm_df[metric] = (norm_df[metric] - norm_df[metric].min()) / (norm_df[metric].max() - norm_df[metric].min())

norm_df["CombinedScore"] = norm_df[metrics].sum(axis=1)
best_combined = norm_df.loc[norm_df["CombinedScore"].idxmax()]

print(f"\n Overall Best Method (based on normalized combined score):")
print(f"  - {best_combined['Method']} (Score: {best_combined['CombinedScore']:.4f})")
