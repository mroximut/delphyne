"""
Visualize aggregate strategy performance heatmap for all_normal_reflect sequence.

Usage:
    python3 visualize_heatmap.py [csv_path]

Default CSV: output_9feb/aggregate_summary.csv
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read CSV file
csv_path = (
    sys.argv[1] if len(sys.argv) > 1 else "output_9feb/aggregate_summary.csv"
)
df = pd.read_csv(csv_path)

# Filter for all_normal_reflect sequence type
df_filtered = df[df["sequence_type"] == "all_normal_reflect"].copy()
df_filtered["accuracy"] = 100 * df_filtered["correct"] / df_filtered["total"]

# Pivot to create heatmap data
# Format: aggregation_type x reflection_type
reflection_order = ["always", "never", "only_if_sat"]
aggregation_order = ["majority_vote", "favor_unsat"]

pivot_data = df_filtered.pivot(
    index="aggregation_type", columns="reflection_type", values="accuracy"
)
data = pivot_data.reindex(
    index=aggregation_order, columns=reflection_order
).values

fig, ax = plt.subplots(figsize=(10, 6))

# Create heatmap
im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=75, vmax=87)

# Set ticks and labels
ax.set_xticks(np.arange(3))
ax.set_yticks(np.arange(2))
ax.set_xticklabels(["Always", "Never", "Only if SAT"])
ax.set_yticklabels(["Majority Vote", "Favor UNSAT"])

ax.set_xlabel("Reflection Type", fontsize=12)
ax.set_ylabel("Aggregation Type", fontsize=12)
ax.set_title(
    "Aggregate Strategy Performance\n(all_normal_reflect sequence)",
    fontsize=14,
    fontweight="bold",
)

# Add text annotations
for i in range(2):
    for j in range(3):
        text = ax.text(
            j,
            i,
            f"{data[i, j]:.1f}%",
            ha="center",
            va="center",
            color="black",
            fontsize=12,
            fontweight="bold",
        )

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Accuracy (%)", rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig(
    "output_9feb/aggregate_heatmap_csv.png", dpi=300, bbox_inches="tight"
)
print(f"Reading data from: {csv_path}")
print("Saved to output_9feb/aggregate_heatmap_csv.png")
print(f"\nBest: {data.max():.2f}%")
print(f"Worst: {data.min():.2f}%")
plt.show()
