"""Compare mixed vs all_normal_reflect sequence types side-by-side.

Usage:
    python3 visualize_mixed_sequence.py [csv_path]

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

# Calculate accuracy
df["accuracy"] = 100 * df["correct"] / df["total"]

# Create side-by-side comparison of sequence types
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ===== Heatmap for all_normal_reflect (left panel) =====
df_reflect = df[df["sequence_type"] == "all_normal_reflect"].copy()
reflection_order = ["always", "never", "only_if_sat"]
aggregation_order = ["majority_vote", "favor_unsat"]

pivot_reflect = df_reflect.pivot(
    index="aggregation_type", columns="reflection_type", values="accuracy"
)
data_reflect = pivot_reflect.reindex(
    index=aggregation_order, columns=reflection_order
).values

im1 = ax1.imshow(data_reflect, cmap="RdYlGn", aspect="auto", vmin=70, vmax=87)

ax1.set_xticks(np.arange(3))
ax1.set_yticks(np.arange(2))
ax1.set_xticklabels(["Always", "Never", "Only if SAT"])
ax1.set_yticklabels(["Majority Vote", "Favor UNSAT"])

ax1.set_xlabel("Reflection Type", fontsize=12)
ax1.set_ylabel("Aggregation Type", fontsize=12)
ax1.set_title("3x Normal Sequence", fontsize=13, fontweight="bold")

# Add text annotations
for i in range(2):
    for j in range(3):
        ax1.text(
            j,
            i,
            f"{data_reflect[i, j]:.1f}%",
            ha="center",
            va="center",
            color="black",
            fontsize=11,
            fontweight="bold",
        )

# Add grid
ax1.set_xticks(np.arange(3) - 0.5, minor=True)
ax1.set_yticks(np.arange(2) - 0.5, minor=True)
ax1.grid(which="minor", color="black", linestyle="-", linewidth=2)
ax1.tick_params(which="minor", size=0)

cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label("Accuracy (%)", rotation=270, labelpad=20)


# ===== Heatmap for mixed sequence (right panel) =====
df_mixed = df[df["sequence_type"] == "mixed"].copy()
reflection_order_mixed = ["mixed", "never", "only_if_sat"]

pivot_mixed = df_mixed.pivot(
    index="aggregation_type", columns="reflection_type", values="accuracy"
)
data_mixed = pivot_mixed.reindex(
    index=aggregation_order, columns=reflection_order_mixed
).values

im2 = ax2.imshow(data_mixed, cmap="RdYlGn", aspect="auto", vmin=70, vmax=87)

ax2.set_xticks(np.arange(3))
ax2.set_yticks(np.arange(2))
ax2.set_xticklabels(["Mixed", "Never", "Only if SAT"])
ax2.set_yticklabels(["Majority Vote", "Favor UNSAT"])

ax2.set_xlabel("Reflection Type", fontsize=12)
ax2.set_ylabel("Aggregation Type", fontsize=12)
ax2.set_title(
    "Literally,Normal,Implicitly Sequence", fontsize=13, fontweight="bold"
)

# Add text annotations
for i in range(2):
    for j in range(3):
        ax2.text(
            j,
            i,
            f"{data_mixed[i, j]:.1f}%",
            ha="center",
            va="center",
            color="black",
            fontsize=11,
            fontweight="bold",
        )

# Add grid
ax2.set_xticks(np.arange(3) - 0.5, minor=True)
ax2.set_yticks(np.arange(2) - 0.5, minor=True)
ax2.grid(which="minor", color="black", linestyle="-", linewidth=2)
ax2.tick_params(which="minor", size=0)

cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label("Accuracy (%)", rotation=270, labelpad=20)

# Overall title
fig.suptitle(
    "Aggregate Strategy Performance by Sequence Type",
    fontsize=15,
    fontweight="bold",
    y=1.00,
)

plt.tight_layout()
plt.savefig(
    "output_9feb/aggregate_sequence_comparison_csv.png",
    dpi=300,
    bbox_inches="tight",
)
print(f"Reading data from: {csv_path}")
print("Saved to output_9feb/aggregate_sequence_comparison_csv.png")
plt.show()

# Calculate statistics dynamically
mixed_max = df_mixed["accuracy"].max()
mixed_min = df_mixed["accuracy"].min()
mixed_max_row = df_mixed.loc[df_mixed["accuracy"].idxmax()]
mixed_min_row = df_mixed.loc[df_mixed["accuracy"].idxmin()]

reflect_max = df_reflect["accuracy"].max()
reflect_min = df_reflect["accuracy"].min()
reflect_max_row = df_reflect.loc[df_reflect["accuracy"].idxmax()]
reflect_min_row = df_reflect.loc[df_reflect["accuracy"].idxmin()]

print("\n=== Mixed vs All Normal Reflect Comparison ===")
print("\nMixed Sequence:")
print(
    f"  Best: {mixed_max_row['aggregation_type']}/{mixed_max_row['reflection_type']} ({mixed_max:.2f}%)"
)
print(
    f"  Worst: {mixed_min_row['aggregation_type']}/{mixed_min_row['reflection_type']} ({mixed_min:.2f}%)"
)
print(f"  Range: {mixed_min:.2f}% - {mixed_max:.2f}%")
print("\nAll Normal Reflect:")
print(
    f"  Best: {reflect_max_row['aggregation_type']}/{reflect_max_row['reflection_type']} ({reflect_max:.2f}%)"
)
print(
    f"  Worst: {reflect_min_row['aggregation_type']}/{reflect_min_row['reflection_type']} ({reflect_min:.2f}%)"
)
print(f"  Range: {reflect_min:.2f}% - {reflect_max:.2f}%")
print("\nKey Insight: All Normal Reflect consistently outperforms Mixed")
print(f"  Overall best difference: {reflect_max - mixed_max:.2f}pp")
