
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot():
    try:
        df = pd.read_csv("benchmark_results.csv")
    except Exception as e:
        print(f"Could not read csv: {e}")
        return

    # Filter B=1 for main trend
    df = df[df['Batch'] == 1]
    
    # Setup styles
    sns.set_style("darkgrid")
    
    # 1. Time (Log Scale) - All Backends
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Size', y='Time_ms', hue='Backend', marker='o')
    plt.yscale('log')
    plt.xscale('log', base=2)
    plt.title('Execution Time vs Matrix Size (Log-Log)')
    plt.ylabel('Time (ms)')
    plt.xlabel('Matrix Size (N)')
    plt.grid(True, which="minor", ls="-", alpha=0.2)
    plt.savefig('benchmark_time_log.png')
    plt.close()
    
    # 2. Time (Linear) - Optimized Only (Torch, Triton, Cpp)
    plt.figure(figsize=(10, 6))
    df_opt = df[df['Backend'].isin(['torch', 'triton', 'cpp_legacy', 'cpp_persistent'])]
    sns.lineplot(data=df_opt, x='Size', y='Time_ms', hue='Backend', marker='o')
    plt.xscale('log', base=2)
    plt.title('Execution Time - GPU Backends')
    plt.ylabel('Time (ms)')
    plt.xlabel('Matrix Size (N)')
    plt.savefig('benchmark_time_linear.png')
    plt.close()
    
    # 3. Memory
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Size', y='Memory_MB', hue='Backend', marker='o')
    plt.xscale('log', base=2)
    plt.ylabel('Peak Memory (MB)')
    plt.xlabel('Matrix Size (N)')
    plt.title('Memory Usage vs Matrix Size')
    plt.savefig('benchmark_memory.png')
    plt.close()
    
    # 4. Speedup Factor (vs Torch)
    plt.figure(figsize=(10, 6))
    pivot = df_opt.pivot(index='Size', columns='Backend', values='Time_ms')
    if 'torch' in pivot.columns:
        if 'triton' in pivot.columns:
            pivot['Speedup_Triton'] = pivot['torch'] / pivot['triton']
            plt.plot(pivot.index, pivot['Speedup_Triton'], marker='o', label='Triton')
            
        if 'cpp_legacy' in pivot.columns:
            pivot['Speedup_CPP_Legacy'] = pivot['torch'] / pivot['cpp_legacy']
            plt.plot(pivot.index, pivot['Speedup_CPP_Legacy'], marker='x', linestyle='--', label='C++ Legacy')
            
        if 'cpp_persistent' in pivot.columns:
            pivot['Speedup_CPP_Persistent'] = pivot['torch'] / pivot['cpp_persistent']
            plt.plot(pivot.index, pivot['Speedup_CPP_Persistent'], marker='^', linestyle='-', label='C++ Persistent')
            
        plt.xscale('log', base=2)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        plt.title('Speedup relative to Torch Backend')
        plt.ylabel('Speedup Factor (x)')
        plt.xlabel('Matrix Size (N)')
        plt.legend()
        plt.savefig('benchmark_speedup.png')
        plt.close()

    print("Graphs generated.")

if __name__ == "__main__":
    plot()
