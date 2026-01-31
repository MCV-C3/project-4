import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import os
import numpy as np

# --- Helper Functions ---

def parse_params(row):
    """Extracts kernel and C value from the params string."""
    try:
        if isinstance(row['params'], str):
            params = ast.literal_eval(row['params'])
            return params.get('kernel'), params.get('C')
        return None, None
    except Exception as e:
        return None, None

def get_kernel_order(df):
    """Returns a consistent order for kernels."""
    preferred = ['linear', 'rbf', 'histogram_intersection']
    available = [k for k in df['kernel'].unique() if pd.notna(k)]
    ordered = [k for k in preferred if k in available] + [k for k in available if k not in preferred]
    return ordered

def layer_sort_key(val):
    """Sorts layers numerically, putting non-numeric layers at the end."""
    if str(val).isdigit():
        return int(val)
    return 999 

def prepare_pivot_for_heatmap(data, metric):
    """Pivots data for heatmap plotting."""
    try:
        pivot_table = data.pivot(index='layer', columns='C', values=metric)
    except ValueError:
        pivot_table = data.pivot_table(index='layer', columns='C', values=metric, aggfunc='mean')
    
    # Sort index (layers)
    sorted_index = sorted(pivot_table.index, key=layer_sort_key)
    pivot_table = pivot_table.reindex(sorted_index)
    
    # Sort columns (C values)
    pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)
    return pivot_table

# --- Data Loading ---

def load_svm_data(paths_dict=None):
    """
    Loads and combines SVM results from CSV files.
    
    Args:
        paths_dict (dict): Dictionary mapping ImgSize (int) to file path (str).
                           If None, uses default paths.
    
    Returns:
        pd.DataFrame: Combined dataframe with 'ImgSize', 'kernel', 'C' columns properly formatted.
    """
    if paths_dict is None:
        paths_dict = {
            16: "/ghome/group04/C3/Benet/project-4/Week2/svm_experiments/SVM_C_sweep_ImgSize_16/results.csv",
            32: "/ghome/group04/C3/Benet/project-4/Week2/svm_experiments/SVM_C_sweep_ImgSize_32/results.csv"
        }

    all_data = []
    for img_size, path in paths_dict.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found")
            continue
        
        try:
            df = pd.read_csv(path)
            df['ImgSize'] = img_size
            
            if 'params' in df.columns:
                df[['kernel', 'C']] = df.apply(parse_params, axis=1, result_type='expand')
            
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    if not all_data:
        print("No data found!")
        return pd.DataFrame()

    df_combined = pd.concat(all_data, ignore_index=True)
    df_combined['layer'] = df_combined['layer'].astype(str)
    df_combined['C'] = pd.to_numeric(df_combined['C'], errors='coerce')
    df_combined.dropna(subset=['kernel', 'C'], inplace=True)
    
    return df_combined

# --- Plotting Functions ---

def plot_heatmap(df, metric_col, metric_name, cmap='viridis', save_path=None, show=True):
    """
    Plots a combined heatmap for all kernels for a given metric.
    """
    img_size = df['ImgSize'].iloc[0] if not df.empty else 'Unknown'
    kernels = get_kernel_order(df)
    if not kernels: return

    # Determine global min/max for colorbar scaling
    vmin = df[metric_col].min()
    vmax = df[metric_col].max()

    fig, axes = plt.subplots(1, len(kernels), figsize=(6 * len(kernels), 6), sharey=True)
    if len(kernels) == 1: axes = [axes] # Handle single subplot case

    # Create a wrapper for colorbar
    cbar_ax = fig.add_axes([.91, .3, .01, .4]) # [left, bottom, width, height]

    for i, (ax, kernel) in enumerate(zip(axes, kernels)):
        kernel_data = df[df['kernel'] == kernel]
        pivot_data = prepare_pivot_for_heatmap(kernel_data, metric_col)
        
        # Only show y-label for the first plot
        ylabel = 'Layer' if i == 0 else ''
        yticklabels = True if i == 0 else False
        
        sns.heatmap(pivot_data, ax=ax, annot=True, cmap=cmap, fmt=".3f", 
                    vmin=vmin, vmax=vmax, cbar= (i == 0), cbar_ax=None if i!=0 else cbar_ax)
        
        ax.set_title(f"{kernel}")
        ax.set_ylabel(ylabel)
        ax.set_xlabel('C Value')
        
    fig.suptitle(f"Heatmap: {metric_name} (ImgSize {img_size})", fontsize=16)
    
    # Handle layout warning by catching it or using try/except if needed, 
    # but usually proper usage of rect avoids it mostly.
    # Note: tight_layout with rect allows room for suptitle/cbar
    with pd.option_context('mode.chained_assignment', None): # Suppress pandas warnings if any
        pass

    if save_path:
        # We need to be careful with tight_layout with cbar_ax manually placed
        # Often it's safer to just save without tight_layout if manually placing, or do it carefully
        # plt.tight_layout(rect=[0, 0, .9, 1]) 
        plt.subplots_adjust(top=0.9, right=0.9)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_lineplot(df, metric_col, std_col, metric_name, save_path=None, show=True):
    """
    Plots a combined line plot with std deviation shading.
    """
    img_size = df['ImgSize'].iloc[0] if not df.empty else 'Unknown'
    kernels = get_kernel_order(df)
    if not kernels: return

    fig, axes = plt.subplots(1, len(kernels), figsize=(6 * len(kernels), 6), sharey=True)
    if len(kernels) == 1: axes = [axes]

    # Get global y limits
    y_vals = df[metric_col]
    y_stds = df[std_col] if std_col in df.columns else 0
    y_min = (y_vals - y_stds).min()
    y_max = (y_vals + y_stds).max()
    margin = (y_max - y_min) * 0.05 if y_max != y_min else 0.1
    y_lim = (max(0, y_min - margin), y_max + margin)
    
    # Cap accuracy at 1.05 for visual breathing room, but not strictly if it's not accuracy
    if 'accuracy' in metric_col:
         y_lim = (max(0, y_min - margin), min(1.05, y_max + margin))

    for i, (ax, kernel) in enumerate(zip(axes, kernels)):
        kernel_data = df[df['kernel'] == kernel].copy()
        
        # Sort layers for consistent colors
        layers = sorted(kernel_data['layer'].unique(), key=layer_sort_key)
        palette = sns.color_palette("tab10", n_colors=len(layers))
        layer_color_map = dict(zip(layers, palette))

        for layer in layers:
            layer_data = kernel_data[kernel_data['layer'] == layer].sort_values('C')
            color = layer_color_map[layer]
            
            # Plot Mean Line
            ax.plot(layer_data['C'], layer_data[metric_col], marker='o', label=layer, color=color)
            
            # Plot Std Deviation area
            if std_col and std_col in layer_data.columns:
                lower = layer_data[metric_col] - layer_data[std_col]
                upper = layer_data[metric_col] + layer_data[std_col]
                ax.fill_between(layer_data['C'], lower, upper, color=color, alpha=0.2)

        ax.set_xscale('log')
        ax.set_title(f"{kernel}")
        ax.set_xlabel('C Value')
        if i == 0:
            ax.set_ylabel(metric_name)
            ax.legend(title='Layer')
        else:
            ax.set_ylabel('')
        
        ax.set_ylim(y_lim)
        ax.grid(True)

    fig.suptitle(f"{metric_name} vs C (ImgSize {img_size})", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        
    if show:
        plt.show()
    else:
        plt.close()

def plot_barplot(df, metric_col, metric_name, save_path=None, show=True):
    """
    Plots a combined bar plot of the best performance per layer.
    """
    img_size = df['ImgSize'].iloc[0] if not df.empty else 'Unknown'
    kernels = get_kernel_order(df)
    if not kernels: return

    # Prepare data: Best C per layer for each kernel
    best_data = []
    for kernel in kernels:
        k_df = df[df['kernel'] == kernel]
        if k_df.empty: continue
        # Find row with max metric per layer
        idx = k_df.groupby('layer')[metric_col].idxmax()
        subset = k_df.loc[idx].copy()
        best_data.append(subset)
    
    if not best_data: return
    df_best = pd.concat(best_data)
    
    # Global Y limits
    y_max = df_best[metric_col].max()
    y_lim = (0, min(1.05, y_max * 1.1)) # Assuming accuracy

    fig, axes = plt.subplots(1, len(kernels), figsize=(6 * len(kernels), 6), sharey=True)
    if len(kernels) == 1: axes = [axes]

    for i, (ax, kernel) in enumerate(zip(axes, kernels)):
        data = df_best[df_best['kernel'] == kernel]
        
        # Sort layers
        data['sort_key'] = data['layer'].apply(layer_sort_key)
        data = data.sort_values('sort_key')
        
        bars = sns.barplot(data=data, x='layer', y=metric_col, ax=ax, palette='viridis', hue='layer', legend=False)
        
        ax.set_title(f"{kernel}")
        ax.set_xlabel('Layer')
        if i == 0:
            ax.set_ylabel(f"Best {metric_name}")
        else:
            ax.set_ylabel('')
        
        ax.set_ylim(y_lim)
        
        # Add values on top
        for container in bars.containers:
            bars.bar_label(container, fmt='%.3f', padding=3)

    fig.suptitle(f"Best {metric_name} per Layer (ImgSize {img_size})", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        
    if show:
        plt.show()
    else:
        plt.close()


# --- Main Execution Block ---

if __name__ == "__main__":
    # This block only runs if the script is executed directly
    
    output_dir = "plots_svm_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    df_combined = load_svm_data()
    
    if df_combined.empty:
        print("No data loaded. Exiting.")
        exit()

    for img_size in [16, 32]:
        subset_img = df_combined[df_combined['ImgSize'] == img_size]
        if subset_img.empty: continue
        
        print(f"Generating plots for ImgSize {img_size}...")
        
        # 1. Heatmaps
        plot_heatmap(
            subset_img, 
            'cv_mean_train_accuracy', 
            'Train Accuracy (CV Mean)', 
            cmap='Blues', 
            save_path=os.path.join(output_dir, f"heatmap_combined_train_acc_img{img_size}.png"),
            show=False
        )
        
        plot_heatmap(
            subset_img, 
            'cv_mean_test_accuracy', 
            'Validation Accuracy (CV Mean)', 
            cmap='viridis', 
            save_path=os.path.join(output_dir, f"heatmap_combined_val_acc_img{img_size}.png"),
            show=False
        )
        
        plot_heatmap(
            subset_img, 
            'mean_execution_time', 
            'Execution Time (s)', 
            cmap='Reds', 
            save_path=os.path.join(output_dir, f"heatmap_combined_time_img{img_size}.png"),
            show=False
        )

        # 2. Line Plots
        plot_lineplot(
            subset_img,
            'cv_mean_test_accuracy',
            'cv_std_test_accuracy',
            'Validation Accuracy',
            save_path=os.path.join(output_dir, f"lineplot_combined_val_acc_img{img_size}.png"),
            show=False
        )

        # 3. Bar Plots
        plot_barplot(
            subset_img,
            'cv_mean_test_accuracy',
            'Validation Accuracy',
            save_path=os.path.join(output_dir, f"barplot_combined_best_val_acc_img{img_size}.png"),
            show=False
        )

    print(f"All plots generated in {output_dir}")
