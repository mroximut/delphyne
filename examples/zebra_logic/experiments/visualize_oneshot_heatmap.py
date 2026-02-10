"""Visualize oneshot and blacklist strategy performance heatmap.

Usage:
    python3 visualize_oneshot_heatmap.py [csv_path]

Default CSV: output_9feb/oneshot_summary.csv
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read CSV file
csv_path = (
    sys.argv[1] if len(sys.argv) > 1 else "output_9feb/oneshot_summary.csv"
)
df = pd.read_csv(csv_path)

# Calculate accuracy
df["accuracy"] = 100 * df["correct"] / df["total"]

# Create mapping for reflection types
reflection_map = {"never": 0, "always": 1, "only_if_sat": 2}
model_map = {
    "literal": 0,
    "implicitly": 1,
    "normal": 2,
    "iterative_blacklist": 3,
}

# Initialize data array with NaN
data = np.full((4, 3), np.nan)

# Fill in the data
for _, row in df.iterrows():
    model = row["oneshot_model_type"]
    reflect = row["oneshot_reflect_type"]
    if model in model_map and reflect in reflection_map:
        data[model_map[model], reflection_map[reflect]] = row["accuracy"]

fig, ax = plt.subplots(figsize=(10, 8))

# Create heatmap with masked values
masked_data = np.ma.masked_invalid(data)
im = ax.imshow(masked_data, cmap="RdYlGn", aspect="auto", vmin=60, vmax=83)

# Set ticks and labels
ax.set_xticks(np.arange(3))
ax.set_yticks(np.arange(4))
ax.set_xticklabels(["No Reflect", "Reflect", "Reflect if SAT"])
ax.set_yticklabels(["Literal", "Implicitly", "Normal", "Iterative\nBlacklist"])

ax.set_xlabel("Reflection Strategy", fontsize=12)
ax.set_ylabel("Model Type / Strategy", fontsize=12)
ax.set_title(
    "Oneshot Strategy Performance (+ Iterative Blacklist)",
    fontsize=14,
    fontweight="bold",
)

# Add text annotations
for i in range(4):
    for j in range(3):
        if not np.isnan(data[i, j]):
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
        else:
            # Mark N/A cells
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=True,
                    facecolor="lightgray",
                    edgecolor="black",
                    linewidth=1,
                )
            )
            ax.text(
                j,
                i,
                "N/A",
                ha="center",
                va="center",
                color="gray",
                fontsize=10,
                style="italic",
            )

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Accuracy (%)", rotation=270, labelpad=20)

# Add grid
ax.set_xticks(np.arange(3) - 0.5, minor=True)
ax.set_yticks(np.arange(4) - 0.5, minor=True)
ax.grid(which="minor", color="black", linestyle="-", linewidth=2)
ax.tick_params(which="minor", size=0)

# Add separator line before iterative blacklist
ax.axhline(y=2.5, color="red", linestyle="--", linewidth=3, alpha=0.7)

plt.tight_layout()
plt.savefig(
    "output_9feb/oneshot_heatmap_csv.png", dpi=300, bbox_inches="tight"
)
print(f"Reading data from: {csv_path}")
print("Saved to output_9feb/oneshot_heatmap_csv.png")
plt.show()

# Print insights
print("\n=== Oneshot Performance Insights ===")
valid_data = data[~np.isnan(data)]
if len(valid_data) > 0:
    print(f"Best configuration: {valid_data.max():.2f}%")
    print(f"Worst configuration: {valid_data.min():.2f}%")

# Calculate stats per model type
for model_name, model_idx in model_map.items():
    model_data = data[model_idx, :]
    valid_model = model_data[~np.isnan(model_data)]
    if len(valid_model) > 0:
        print(
            f"\n{model_name.capitalize()}: {valid_model.min():.2f}% - {valid_model.max():.2f}% (avg: {valid_model.mean():.2f}%)"
        )
