import os
import pandas as pd
import matplotlib.pyplot as plt

# List of your sensitivity result CSV files
filenames = [
    "sensitivity_k_results_coauthorship_cora.csv",
    "sensitivity_k_results_coauthorship_dblp.csv",
    "sensitivity_k_results_cocitation_citeseer.csv",
    "sensitivity_k_results_cocitation_cora.csv",
    "sensitivity_k_results_npz_20news.csv",
    "sensitivity_k_results_npz_query.csv"
]

# Where to save plots and summary
output_dir = "sensitivity_output"
os.makedirs(output_dir, exist_ok=True)

# Collect best results per dataset
summary_rows = []

# Custom x-ticks to match AHCKA paper
custom_ticks = [2, 50, 100, 500, 1000]

for file in filenames:
    df = pd.read_csv(file)
    dataset = file.replace("sensitivity_k_results_", "").replace(".csv", "")

    # 🖼️ Plotting
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["k"], df["Accuracy"], 'b--', label='Acc')
    ax1.plot(df["k"], df["F1-score"], color='orange', linestyle=':', label='F1')
    ax1.plot(df["k"], df["NMI"], 'g-', label='NMI')

    ax1.set_xlabel("k", fontsize=12)
    ax1.set_ylabel("Clustering Score", fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.set_title(f"Impact of k on AHCKA Clustering ({dataset})", fontsize=14)

    ax1.set_xticks(custom_ticks)
    ax1.set_xticklabels([str(k) for k in custom_ticks], fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(df["k"], df["Runtime"], 'm-', label='time')
    ax2.set_ylabel("Time (s)", color='m', fontsize=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ahcka_sensitivity_{dataset}.png", dpi=300)
    plt.close()

    # 📊 Best row based on Accuracy
    best_row = df.loc[df["Accuracy"].idxmax()]
    summary_rows.append({
        "Dataset": dataset,
        "Best k": int(best_row["k"]),
        "Accuracy": round(best_row["Accuracy"], 3),
        "F1-score": round(best_row["F1-score"], 3),
        "NMI": round(best_row["NMI"], 3),
        "Runtime (s)": round(best_row["Runtime"], 2),
        "Memory (MB)": round(best_row["Memory (MB)"], 2)
    })

# 💾 Save summary
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{output_dir}/ahcka_summary_table.csv", index=False)

print("✅ All plots and summary table generated in 'sensitivity_output/' folder.")
