
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# --- Configuration ---
CSV_PATH = "benchmark_exhaustive_results.csv"
OUTPUT_DIR = "exhaustive_plots_detailed"
BACKEND_PALETTE = {
    "torch": "#1f77b4",          # Blue
    "torch_compiled": "#ff7f0e", # Orange
    "triton": "#d62728",         # Red
    "cuda": "#2ca02c",           # Green
    "cutlass": "#9467bd",        # Purple
    "cpp": "#8c564b"             # Brown
}
BACKEND_MARKERS = {
    "torch": "o",
    "torch_compiled": "X",
    "triton": "D",
    "cuda": "s",
    "cutlass": "v",
    "cpp": "^"
}

def plot_detailed():
    if not os.path.exists(CSV_PATH):
        print(f"CSV {CSV_PATH} not found.")
        return

    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    
    # Analyze Failures
    failures = df[df["Status"] != "Success"]
    if not failures.empty:
        print("\n--- Failure Summary ---")
        print(failures.groupby(["Algo", "Backend", "Status"]).size())
        print("-----------------------\n")

    # Filter only Success
    df = df[df["Status"] == "Success"].copy()
    if df.empty:
        print("No success data found.")
        return

    # 1. Feature Engineering
    # distinct label for N x M
    df["Config"] = df["Rows"].astype(str) + "x" + df["Cols"].astype(str)
    df["ProblemSize"] = df["Rows"] * df["Cols"]
    
    # Ensure directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set Style
    sns.set_theme(style="whitegrid", context="talk") # "talk" makes everything larger/readable
    
    # 2. Iteration Strategy: Algo -> Precision -> Batch
    # We create one BIG distinct file for each of these combos.
    
    unique_algos = df["Algo"].unique()
    unique_precs = df["Precision"].unique()
    unique_batches = df["Batch"].unique()
    
    total_plots = len(unique_algos) * len(unique_precs) * len(unique_batches)
    curr = 0
    
    print(f"Generating {total_plots} detailed reports...")
    
    for algo in unique_algos:
        for prec in unique_precs:
            for batch in unique_batches:
                # Filter Slice
                subset = df[
                    (df["Algo"] == algo) & 
                    (df["Precision"] == prec) & 
                    (df["Batch"] == batch)
                ].copy()
                
                if subset.empty:
                    continue
                
                # Sort by Problem Size explicitly so X-axis is intuitive
                subset = subset.sort_values(by=["ProblemSize", "Rows", "Cols"])
                
                # Create the Figure: 3 Subplots (Mean, Std, VRAM)
                # We do NOT average over n_rows/n_cols. They are the X-axis categories.
                
                fig, axes = plt.subplots(3, 1, figsize=(16, 18), sharex=True)
                fig.suptitle(f"{algo} | {prec} | Batch={batch}", fontsize=20, y=0.95)
                
                # --- Subplot 1: Time Mean (Log Scale) ---
                ax1 = axes[0]
                sns.lineplot(
                    data=subset, x="Config", y="Time Mean", hue="Backend", style="Backend",
                    markers=BACKEND_MARKERS, dashes=False, palette=BACKEND_PALETTE,
                    linewidth=3, markersize=12, ax=ax1
                )
                ax1.set_yscale("log")
                ax1.set_ylabel("Runtime Mean (ms) [Log]")
                ax1.grid(True, which="minor", ls=":", alpha=0.5)
                ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
                ax1.set_title("Performance (Lower is Better)")

                # --- Subplot 2: Time Std (Stability) ---
                ax2 = axes[1]
                sns.lineplot(
                    data=subset, x="Config", y="Time Std", hue="Backend", style="Backend",
                    markers=BACKEND_MARKERS, dashes=False, palette=BACKEND_PALETTE,
                    linewidth=3, markersize=12, ax=ax2, legend=False
                )
                ax2.set_ylabel("Runtime Std Dev (ms)")
                # ax2.set_yscale("log") # Std dev might be 0, linear is often safer or log1p
                # Let's try Linear for Std unless range is huge.
                ax2.set_title("Stability / Variance (Lower is Better)")

                # --- Subplot 3: VRAM Max ---
                ax3 = axes[2]
                sns.lineplot(
                    data=subset, x="Config", y="VRAM Max", hue="Backend", style="Backend",
                    markers=BACKEND_MARKERS, dashes=False, palette=BACKEND_PALETTE,
                    linewidth=3, markersize=12, ax=ax3, legend=False
                )
                ax3.set_ylabel("Peak VRAM (MB)")
                ax3.set_xlabel("Matrix Size (Rows x Cols)")
                ax3.set_title("Memory Efficiency (Lower is Better)")
                
                # Layout adjustments
                # Rotate X-labels if many
                plt.xticks(rotation=45, ha='right')
                
                plt.tight_layout(rect=[0, 0, 0.85, 0.95]) # Make room for legend and title
                
                # Save
                fname = f"{algo}_{prec}_B{batch}_detailed.png"
                save_path = os.path.join(OUTPUT_DIR, fname)
                plt.savefig(save_path, dpi=100)
                plt.close()
                
                print(f"Generated: {fname}")
                curr += 1

    print(f"Done. Check {OUTPUT_DIR}/ for results.")

if __name__ == "__main__":
    plot_detailed()
