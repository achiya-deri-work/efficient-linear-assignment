
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
CSV_PATH = "benchmark_exhaustive_results.csv"
OUTPUT_DIR = "exhaustive_plots_comparative"
BACKEND_PALETTE = {
    "torch": "#1f77b4",          # Blue (Standard)
    "torch_compiled": "#ff7f0e", # Orange (Standard)
    "triton": "#000000",         # Black (High Contrast against Orange/Blue)
    "cuda": "#2ca02c",           # Green (Standard)
    "cutlass": "#e377c2",        # Pink/Magenta (Distinct from Orange/Red)
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

def load_data():
    if not os.path.exists(CSV_PATH):
        print(f"CSV {CSV_PATH} not found.")
        return None
    df = pd.read_csv(CSV_PATH)
    # Filter Success
    df = df[df["Status"] == "Success"].copy()
    
    # Feature Engineering
    df["ProblemSize"] = df["Rows"] * df["Cols"]
    return df

def plot_comparison_dtype(df):
    """
    Compares different precisions for the SAME Algorithm + Backend.
    """
    print("Generating DType Comparisons...")
    
    unique_algos = df["Algo"].unique()
    unique_backends = df["Backend"].unique()
    
    for algo in unique_algos:
        for backend in unique_backends:
            subset = df[(df["Algo"] == algo) & (df["Backend"] == backend)].copy()
            if subset.empty: continue
            
            # We want to see Precision scaling
            # Hue = Precision
            
            # Sort
            subset = subset.sort_values(by=["ProblemSize", "Rows", "Cols"])
            subset["Config"] = subset["Rows"].astype(str) + "x" + subset["Cols"].astype(str)
            
            plt.figure(figsize=(14, 8))
            sns.lineplot(
                data=subset, x="Config", y="Time Mean", hue="Precision", style="Precision",
                markers=True, dashes=False, linewidth=3, markersize=10
            )
            plt.yscale("log")
            plt.title(f"Precision Impact: {algo} ({backend})")
            plt.ylabel("Time (ms) [Log]")
            plt.grid(True, which="minor", ls=":", alpha=0.5)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            fname = f"Compare_Prec_{algo}_{backend}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, fname))
            plt.close()

def plot_comparison_algo(df):
    """
    Compares different Algorithms for the SAME Precision + Backend.
    """
    print("Generating Algorithm Comparisons...")
    
    unique_precs = df["Precision"].unique()
    unique_backends = df["Backend"].unique()
    
    for prec in unique_precs:
        for backend in unique_backends:
            subset = df[(df["Precision"] == prec) & (df["Backend"] == backend)].copy()
            if subset.empty: continue
            
            subset = subset.sort_values(by=["ProblemSize", "Rows", "Cols"])
            subset["Config"] = subset["Rows"].astype(str) + "x" + subset["Cols"].astype(str)
            
            plt.figure(figsize=(14, 8))
            sns.lineplot(
                data=subset, x="Config", y="Time Mean", hue="Algo", style="Algo",
                markers=True, dashes=False, linewidth=3, markersize=10
            )
            plt.yscale("log")
            plt.title(f"Algorithm Comparison: {backend} ({prec})")
            plt.ylabel("Time (ms) [Log]")
            plt.grid(True, which="minor", ls=":", alpha=0.5)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            fname = f"Compare_Algo_{backend}_{prec}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, fname))
            plt.close()

def plot_3d_surfaces(df):
    """
    3D Plots: X=Rows, Y=Metric, Z=Cols (Visualized as 3D surface).
    Actually Matplotlib 3D: X=Rows, Y=Cols, Z=Metric.
    """
    print("Generating 3D Surface Plots...")
    
    unique_algos = df["Algo"].unique()
    unique_precs = df["Precision"].unique()
    # Filter for relevant backends to reduce clutter? Or plot best backend?
    # Let's plot 'torch_compiled' vs 'triton' vs 'cuda' in separate plots per Algo/Prec.
    
    # We need a grid.
    unique_rows = sorted(df["Rows"].unique())
    unique_cols = sorted(df["Cols"].unique())
    X, Y = np.meshgrid(unique_rows, unique_cols)
    
    for0_algos = df["Algo"].unique()
    for algo in for0_algos:
        for prec in unique_precs:
            # We want to plot multiple backends on one 3D plot? No, too messy.
            # One plot per (Algo, Precision), showing surfaces for different backends.
            
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            plotted_any = False
            
            # Filter Backends
            backends_to_plot = ["torch_compiled", "triton", "cuda", "cutlass"]
            
            for backend in backends_to_plot:
                subset = df[
                    (df["Algo"] == algo) & 
                    (df["Precision"] == prec) & 
                    (df["Backend"] == backend)
                ]
                if subset.empty: continue
                
                # Z Grid
                Z = np.zeros_like(X, dtype=float)
                for i, r in enumerate(unique_rows):
                    for j, c in enumerate(unique_cols):
                        val = subset[(subset["Rows"] == r) & (subset["Cols"] == c)]["Time Mean"]
                        if not val.empty:
                            Z[j, i] = val.values[0] # Note meshgrid indexing
                        else:
                            Z[j, i] = np.nan
                
                # Plot Surface
                try:
                    surf = ax.plot_surface(X, Y, Z, alpha=0.7, label=backend, color=BACKEND_PALETTE.get(backend, 'gray'), antialiased=True)
                    # Proxy artist
                    surf._facecolors2d = surf._facecolor3d
                    surf._edgecolors2d = surf._edgecolor3d
                    plotted_any = True
                except Exception as e:
                    print(f"Skipping surface {backend}: {e}")

            if plotted_any:
                ax.set_xlabel('Rows')
                ax.set_ylabel('Cols')
                ax.set_zlabel('Time (ms)')
                ax.set_title(f"3D Performance Surface: {algo} ({prec})")
                
                # ROTATION: View from low corner looking up slope ("climb inwards")
                ax.view_init(elev=25, azim=-120)
                
                # Legend
                # Custom legend
                import matplotlib.patches as mpatches
                patches = [mpatches.Patch(color=BACKEND_PALETTE[b], label=b, alpha=0.5) for b in backends_to_plot if b in df["Backend"].values]
                ax.legend(handles=patches)
                
                fname = f"3D_Surface_{algo}_{prec}.png"
                plt.savefig(os.path.join(OUTPUT_DIR, fname))
                plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    if df is None or df.empty:
        return

    plot_comparison_dtype(df)
    plot_comparison_algo(df)
    plot_3d_surfaces(df)
    
    print(f"Done. Check {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
