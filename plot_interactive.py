import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import numpy as np

# --- Configuration ---
CSV_PATH = "benchmark_exhaustive_results.csv"
OUTPUT_DIR = "interactive_plots"

# Consistent Colors
BACKEND_COLORS = {
    "torch": "blue",
    "torch_compiled": "orange",
    "triton": "red",
    "cuda": "green",
    "cutlass": "purple",
    "cpp": "brown",
    "triton_implicit": "black"
}

def load_data():
    if not os.path.exists(CSV_PATH):
        print(f"CSV {CSV_PATH} not found.")
        return None
    df = pd.read_csv(CSV_PATH)
    return df[df["Status"] == "Success"].copy()

def generate_interactive_plots():
    df = load_data()
    if df is None: return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    unique_algos = df["Algo"].unique()
    unique_precs = df["Precision"].unique()
    
    # Get Grid Dimensions
    unique_rows = sorted(df["Rows"].unique())
    unique_cols = sorted(df["Cols"].unique())
    
    print(f"Generating {len(unique_algos) * len(unique_precs)} interactive html plots...")

    for algo in unique_algos:
        for prec in unique_precs:
            
            fig = go.Figure()
            title = f"Performance Surface: {algo} ({prec})"
            
            # Filter
            subset_base = df[(df["Algo"] == algo) & (df["Precision"] == prec)]
            if subset_base.empty: continue
            
            backends = subset_base["Backend"].unique()
            
            for backend in backends:
                subset = subset_base[subset_base["Backend"] == backend]
                
                # Create Grid
                # Pivot: Rows as Index, Cols as Columns.
                try:
                    pivoted = subset.pivot(index="Rows", columns="Cols", values="Time Mean")
                    
                    # Reindex to ensure full grid (handle missing data with NaNs)
                    pivoted = pivoted.reindex(index=unique_rows, columns=unique_cols)
                    
                    z_data = pivoted.values
                    
                    # Plotly Surface expects Z as 2D array. X and Y as 1D or 2D.
                    # X = Cols (columns), Y = Rows (index) to match matrix layout? 
                    # Usually x=rows, y=cols in 3D plot logic.
                    
                    fig.add_trace(go.Surface(
                        z=z_data,
                        x=unique_cols, # Cols on X axis
                        y=unique_rows, # Rows on Y axis
                        name=backend,
                        showscale=False, # Disable individual colorbars
                        opacity=0.8,
                        colorscale=[[0, BACKEND_COLORS.get(backend, 'gray')], [1, BACKEND_COLORS.get(backend, 'gray')]], # Monochromatic surface
                        hovertemplate=f"<b>{backend}</b><br>Rows: %{{y}}<br>Cols: %{{x}}<br>Time: %{{z:.2f}} ms<extra></extra>"
                    ))
                    
                    # Add Wireframe/Scatter points for clarity?
                    # Surface sometimes hides specific points.
                    # Scatter3d is good for exact points.
                    fig.add_trace(go.Scatter3d(
                        x=subset["Cols"],
                        y=subset["Rows"],
                        z=subset["Time Mean"],
                        mode='markers',
                        marker=dict(size=3, color=BACKEND_COLORS.get(backend, 'gray')),
                        name=f"{backend} (pts)",
                        showlegend=False,
                         hovertemplate=f"<b>{backend}</b><br>Rows: %{{y}}<br>Cols: %{{x}}<br>Time: %{{z:.2f}} ms<extra></extra>"
                    ))

                except Exception as e:
                    print(f"Error plotting {algo} {prec} {backend}: {e}")

            # Layout Update
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='Columns (M)',
                    yaxis_title='Rows (N)',
                    zaxis_title='Runtime (ms)',
                    zaxis_type="log" # Log scale is crucial for large variance
                ),
                width=1200,
                height=900,
                margin=dict(l=65, r=50, b=65, t=90)
            )
            
            filename = f"{algo}_{prec}_interactive.html"
            save_path = os.path.join(OUTPUT_DIR, filename)
            fig.write_html(save_path)
            print(f"Saved {save_path}")

if __name__ == "__main__":
    generate_interactive_plots()
