import os
import re
import cv2
import json
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.ticker import ScalarFormatter
from main import run_experiment 
from bovw import BOVW
from PIL import Image

# -------------------------------------------------------------------------
# Experiment Helpers
# -------------------------------------------------------------------------

def load_json(config_path):
    """
    Loads the experiments list from a json file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Please create the file {config_path} first!")

    with open(config_path, "r") as f:
        experiments_list = json.load(f)

    return experiments_list

def load_existing_results(csv_path):
    """
    Checks if a results CSV exists. If so, loads it as a list of dictionaries.
    If not, returns an empty list.
    """
    if os.path.exists(csv_path):
        print(f"Resuming from existing file: {csv_path}")
        df = pd.read_csv(csv_path)
        return df.to_dict("records")
    return []

def is_experiment_completed(run_name, results_list, key="Descriptor"):
    """
    Checks if a specific run_name already exists in the results list.
    """
    return any(d[key] == run_name for d in results_list)

def run_experiments_from_config(config_path, results_path):
    """
    Orchestrates the running of experiments defined in a config JSON.
    Loads config, checks for existing results, runs missing experiments,
    and saves progress to a CSV.
    """
    # Load experiments and existing results
    experiments_list = load_json(config_path=config_path)
    results_data = load_existing_results(results_path)
    
    print(f"Loaded {len(experiments_list)} experiments from {config_path}")
    
    # Filter out completed experiments
    experiments_to_run = [
        exp for exp in experiments_list 
        if not is_experiment_completed(exp["name"], results_data)
    ]
    print(f"{len(experiments_to_run)} experiments to run after filtering completed ones.")
    
    for exp in experiments_to_run:
        run_name = exp["name"]
        cfg = exp["config"]
        
        print(f"\n==========================================")
        print(f"Running: {run_name}")
        print(f"==========================================")
        
        start_time = time.time()
        
        # Execute (assumes run_experiment returns train_acc, test_acc, cv_scores)
        train_acc, test_acc, cv_scores = run_experiment(cfg)
        
        duration = time.time() - start_time
        
        results_data.append({
            "Descriptor": run_name,
            "Train Accuracy": train_acc,
            "Test Accuracy": test_acc,
            "CV Scores": cv_scores,
            "Time (s)": duration
        })
        
        # Save intermediate results
        if len(results_data) > 0:
            df = pd.DataFrame(results_data)
            df.to_csv(results_path, index=False)

    # Final display and save
    if len(results_data) > 0:
        df = pd.DataFrame(results_data)
        print("Processed Results:")
        # In a script we might not want display(), but for notebook usage usually fine.
        # We'll just return the df so the notebook can display it.
        return df
    else:
        return pd.DataFrame()

# -------------------------------------------------------------------------
# Plotting Helpers
# -------------------------------------------------------------------------

def _plot_grouped_bar(df, x_col, hue_col, title, ylim=None):
    """
    Helper for grouped bar plots (Train vs CV) with fixed label positions.
    """
    plt.figure(figsize=(10, 6))
    
    # 1. Define the specific order of bars
    bar_order = ["Train Accuracy", "CV Mean"]
    
    # 2. Melt for plotting
    df_melted = df.melt(
        id_vars=[x_col], 
        value_vars=bar_order, 
        var_name=hue_col, 
        value_name="Accuracy"
    )
    
    # 3. Plot with explicit hue_order
    ax = sns.barplot(
        x=x_col,
        y="Accuracy",
        hue=hue_col,
        data=df_melted,
        palette="viridis",
        hue_order=bar_order
    )
    
    # 4. Add labels for Train Accuracy bars only
    # We use bar_label for the first container (Train) as it's simple
    ax.bar_label(ax.containers[0], fmt='%.3f', padding=3, fontsize=10)

    # 5. Add Error Bars and MANUAL Labels to the CV Mean bars (container 1)
    if 'CV Std' in df.columns:
        cv_bars = ax.containers[1] # Index 1 corresponds to "CV Mean"
        
        # Iterate through the bars, their std dev, and their mean value
        for bar, std_dev, value in zip(cv_bars, df['CV Std'], df['CV Mean']):
            x_pos = bar.get_x() + bar.get_width() / 2
            y_pos = bar.get_height()
            
            # Draw the error bar
            ax.errorbar(
                x_pos, y_pos, yerr=std_dev, 
                fmt='none', color='black', capsize=5, elinewidth=1.5
            )
            
            # --- THIS IS THE FIX ---
            # Manually place the text above the error bar
            # The y-position is the mean (y_pos) + the std dev + padding
            label_y_pos = y_pos + std_dev + 0.01 
            plt.text(x_pos, label_y_pos, f'{value:.3f}', 
                     ha='center', va='bottom', fontsize=10, color='black')

    # 6. Overfitting lines logic (Train vs CV Mean)
    containers = ax.containers
    if len(containers) >= 2:
        train_bars = containers[0]
        cv_bars = containers[1]
        
        # Zip in std_dev to calculate the correct height for the gap text
        for train_bar, cv_bar, std_dev in zip(train_bars, cv_bars, df['CV Std']):
            x1 = train_bar.get_x() + train_bar.get_width() / 2
            y1 = train_bar.get_height()
            x2 = cv_bar.get_x() + cv_bar.get_width() / 2
            y2 = cv_bar.get_height()
            
            gap = y1 - y2
            mid_x = (x1 + x2) / 2
            
            # Place the text above the highest point of either bar+error
            highest_point = max(y1, y2 + std_dev)
            mid_y = highest_point + 0.04 # Add padding for the gap text line
            
            plt.plot([x1, x2], [y1, y2], color='#c44e52', linestyle='--', marker='o', linewidth=1.5, markersize=4)
            plt.text(mid_x, mid_y, f"+{gap:.3f}", color='#c44e52', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 7. Final Adjustments
    plt.title(title, fontsize=16)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(title="Metric")
    
    # Automatically increase y-limit if labels are cut off
    ymin, ymax = plt.ylim()
    max_y_data = max(df['Train Accuracy'].max(), (df['CV Mean'] + df['CV Std']).max())
    required_ymax = max_y_data + 0.15 # Add sufficient padding for all labels
    
    if ylim is None or required_ymax > ylim[1]:
        plt.ylim(ymin, required_ymax)
    else:
         plt.ylim(ylim)

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_local_descriptors(df):
    """
    Plots results for Local Descriptors (SIFT, ORB, AKAZE) using CV scores.
    """
    # --- Step A: Pre-process the CV Scores column ---
    # Convert string "[0.25 0.27 ...]" to actual numpy arrays
    if isinstance(df['CV Scores'].iloc[0], str):
        # Helper to parse the specific string format
        def parse_cv(s):
            s = s.replace('[', '').replace(']', '')
            return np.array([float(x) for x in s.split()])
            
        cv_arrays = df['CV Scores'].apply(parse_cv)
    else:
        cv_arrays = df['CV Scores']

    # Create the columns needed for the plot
    df['CV Mean'] = cv_arrays.apply(np.mean)
    df['CV Std'] = cv_arrays.apply(np.std)

    # --- Step B: Call the plotting function ---
    _plot_grouped_bar(
        df, 
        x_col="Descriptor", 
        hue_col="Split", 
        title="Local Descriptors Performance (CV)", 
        ylim=(0, 0.40) # Increased slightly to accommodate error bars
    )


def plot_sift_nfeatures(df):
    """
    Plots results for SIFT NFeatures using CV Scores (Mean + Std Dev band).
    """
    df = df.copy()

    # 1. Parse CV Scores
    # Convert string "[0.2 0.3]" -> numpy array
    def parse_cv(s):
        s = s.replace('[', '').replace(']', '')
        return np.array([float(x) for x in s.split()])
    
    cv_arrays = df['CV Scores'].apply(parse_cv)
    df['CV Mean'] = cv_arrays.apply(np.mean)
    df['CV Std'] = cv_arrays.apply(np.std)

    # 2. Extract Integers for the Continuous X-Axis
    df['n_features'] = df['Descriptor'].str.extract(r'(\d+)').astype(int)
    
    # Sort by n_features to ensure lines connect correctly
    df = df.sort_values('n_features')

    plt.figure(figsize=(12, 6))

    # 3. Plot Train Accuracy (Standard Line)
    plt.plot(df['n_features'], df['Train Accuracy'], 
             marker='o', markersize=6, linewidth=2, 
             label='Train Accuracy', color='#2b7bba') # Blue-ish

    # 4. Plot CV Accuracy (Line + Shaded Region)
    # The Mean Line
    plt.plot(df['n_features'], df['CV Mean'], 
             marker='o', markersize=6, linewidth=2, 
             label='CV Accuracy (Mean)', color='#5aa154') # Green-ish
    
    # The Standard Deviation Band (Confidence Interval)
    plt.fill_between(df['n_features'], 
                     df['CV Mean'] - df['CV Std'], 
                     df['CV Mean'] + df['CV Std'], 
                     color='#5aa154', alpha=0.2, label='CV Std Dev')

    # 5. Intelligent Label Placement
    # Train Labels: Just above the point
    for x, y in zip(df['n_features'], df['Train Accuracy']):
        plt.text(x, y + 0.003, f'{y:.3f}', ha='center', va='bottom', fontsize=9, color='#2b7bba', fontweight='bold')

    # CV Labels: Above the Shaded Region (Mean + Std + padding)
    for x, y, std in zip(df['n_features'], df['CV Mean'], df['CV Std']):
        # Position label above the top of the error band
        label_y = y + std + 0.003
        plt.text(x, label_y, f'{y:.3f}', ha='center', va='bottom', fontsize=9, color='#5aa154', fontweight='bold')

    # 6. Formatting
    plt.title("SIFT: Number of Features Performance (Cross-Validation)", fontsize=14)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Number of Features (Log Scale)", fontsize=12)
    
    # Adjust Y-limits to fit labels
    # We look at the max height of (Train) vs (CV + Std)
    max_height = max(df['Train Accuracy'].max(), (df['CV Mean'] + df['CV Std']).max())
    plt.ylim(0.15, max_height + 0.03) 

    # Log Scale for X-Axis
    plt.xscale('log')
    
    # Custom Ticks
    tick_vals = df['n_features'].unique()
    plt.xticks(tick_vals, tick_vals)
    
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    

def plot_dense_sift_step_size_scale(df):
    """
    Plots results for Dense SIFT Step Size & Scale.
    Parses 'Descriptor' to get 'Step Size' and 'Scale Factor'.
    """
    df = df.copy()

    def parse_cv(s):
        s = s.replace('[', '').replace(']', '')
        return np.array([float(x) for x in s.split()])
    
    cv_arrays = df['CV Scores'].apply(parse_cv)
    df['CV Mean'] = cv_arrays.apply(np.mean)
    df['CV Std'] = cv_arrays.apply(np.std)
    
    def parse_desc(desc):
        # Expected format: "DenseSIFT_Step<step>_Scale<scale>"
        s_match = re.search(r'Step(\d+)', desc)
        sc_match = re.search(r'Scale(\d+)', desc)
        step = int(s_match.group(1)) if s_match else 0
        scale = int(sc_match.group(1)) if sc_match else 0
        return pd.Series([step, scale])
    
    # --- PLOT ---
    plt.figure(figsize=(12, 6))

    # Ensure X-axis is treated as categorical (strings) to avoid uneven spacing
    df["Step Size"] = df["Step Size"].astype(str)

    # Grouped Bar Plot by Scale Factor
    ax = sns.barplot(
        data=df,
        x="Step Size",
        y="CV Mean",
        hue="Scale Factor",
        palette="viridis"
    )

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

    plt.title("Dense SIFT: Step Size and Scale")
    plt.ylim(0, 0.6) # Adjust this limit based on your max results
    plt.ylabel("CV Mean Accuracy")
    plt.xlabel("Step Size (px)")
    plt.legend(title="Scale Factor", loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.show()

def plot_dense_sift_times(df):
    """
    Plots times for Dense SIFT Step Size & Scale.
    """
    df = df.copy()

    # 2. PLOTTING
    plt.figure(figsize=(12, 6))

    # Ensure X-axis is treated as categorical so bars are spaced evenly
    df["Step Size"] = df["Step Size"].astype(str)

    # Grouped Bar Plot
    ax = sns.barplot(
        data=df,
        x="Step Size",
        y="Time (s)",
        hue="Scale Factor",
        palette="viridis",
        edgecolor="black"
    )

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=9)

    plt.title("Dense SIFT: Computation Time by Step Size and Scale Factor\n(Lower is Better)")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Step Size (px)")
    plt.legend(title="Scale Factor", loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_codebook_size(df):
    """
    Plots results for Codebook Size experiments using FacetGrid.
    Compares Train Accuracy vs CV Accuracy (Mean ± Std).
    """
    df = df.copy()
    
    # --- 1. Data Parsing ---
    
    # A. Parse Method and K from 'Descriptor'
    def parse_row(row):
        s = str(row)
        # Extract k value
        k_match = re.search(r'k=(\d+)', s)
        k = int(k_match.group(1)) if k_match else 0
        
        # Extract Method Name (Clean up the string)
        # SIFT (k=10) -> SIFT
        # Dense SIFT 8x8 (k=10) -> Dense SIFT 8x8
        if '(' in s:
            name = s.split('(')[0].strip()
        elif 'k=' in s:
            name = s.split('k=')[0].strip().rstrip('_') # handle _k= case
        else:
            name = s
        return pd.Series([name, k])

    df[['Method', 'Codebook_Size']] = df['Descriptor'].apply(parse_row)

    # B. Parse CV Scores
    def parse_cv(s):
        s = str(s).replace('[', '').replace(']', '')
        return np.array([float(x) for x in s.split()])
    
    cv_arrays = df['CV Scores'].apply(parse_cv)
    df['CV Mean'] = cv_arrays.apply(np.mean)
    df['CV Std'] = cv_arrays.apply(np.std)

    # Sort for correct line plotting
    df = df.sort_values(by=["Method", "Codebook_Size"])

    # --- 2. Custom Plotting Function for FacetGrid ---
    def plot_mean_std_lines(data, **kwargs):
        # We access the current subplot's ax
        ax = plt.gca()
        
        # Plot Train (Blue Line)
        ax.plot(
            data['Codebook_Size'], data['Train Accuracy'], 
            marker='o', markersize=6, linewidth=2, 
            color='#2b7bba', label='Train Accuracy'
        )
        
        # Plot CV Mean (Green Line)
        ax.plot(
            data['Codebook_Size'], data['CV Mean'], 
            marker='o', markersize=6, linewidth=2, 
            color='#5aa154', label='CV Accuracy (Mean)'
        )
        
        # Plot CV Confidence Band (Shaded)
        ax.fill_between(
            data['Codebook_Size'], 
            data['CV Mean'] - data['CV Std'], 
            data['CV Mean'] + data['CV Std'], 
            color='#5aa154', alpha=0.2
        )

        # --- Intelligent Labels ---
        for i, row in data.iterrows():
            k = row['Codebook_Size']
            
            # Label Train: Above the point
            train_val = row['Train Accuracy']
            ax.text(k, train_val + 0.015, f'{train_val:.3f}', 
                    ha='center', va='bottom', fontsize=8, color='#2b7bba')
            
            # Label CV: Below the confidence band
            cv_mean = row['CV Mean']
            cv_std = row['CV Std']
            lower_bound = cv_mean - cv_std
            ax.text(k, lower_bound - 0.015, f'{cv_mean:.3f}', 
                    ha='center', va='top', fontsize=8, color='#5aa154')

    # --- 3. Setup FacetGrid ---
    g = sns.FacetGrid(df, col="Method", height=6, aspect=1.1, sharey=True)
    
    # Map the custom function
    g.map_dataframe(plot_mean_std_lines)

    # --- 4. Styling ---
    unique_k = sorted(df['Codebook_Size'].unique())

    for ax in g.axes.flat:
        ax.set_xscale('log')
        ax.set_xticks(unique_k)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel("Codebook Size k (log scale)", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        
        # Adjust Y-limits to fit the new labels and bands
        # Find global max/min for robustness, or use fixed range as requested
        # Using a slightly wider range than 0.15-0.6 to fit text labels safely
        ax.set_ylim(0.1, 0.65)

    # Add a single Legend (Create proxy handles since map_dataframe doesn't auto-legend well)
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    
    legend_elements = [
        Line2D([0], [0], color='#2b7bba', lw=2, marker='o', label='Train Accuracy'),
        Line2D([0], [0], color='#5aa154', lw=2, marker='o', label='CV Accuracy (Mean)'),
        mpatches.Patch(color='#5aa154', alpha=0.2, label='CV Std Dev Region')
    ]
    
    # Place legend on the last axes or outside
    plt.legend(handles=legend_elements, loc='upper left', title="Metric", frameon=True)
    
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle("Codebook Size Comparison: Train vs CV Performance", fontsize=16)
    
    plt.show()


def plot_dim_reduction(df):
    """
    Plots results for Dimensionality Reduction (PCA) experiments.
    Compares Train Accuracy vs CV Accuracy (Mean ± Std).
    """
    df = df.copy()
    
    # --- 1. Data Parsing ---
    
    # A. Parse Method and Components from 'Descriptor'
    def parse_pca_row(row):
        s = str(row)
        # Extract number from "pca_components=X" or similar
        n_match = re.search(r'components=(\d+)', s)
        # Fallback if just a number is present
        if not n_match: 
             n_match = re.search(r'(\d+)', s)
             
        n_val = int(n_match.group(1)) if n_match else 0
        
        # Extract Method Name
        if '(' in s:
            name = s.split('(')[0].strip()
        else:
            name = s
        return pd.Series([name, n_val])

    df[['Method', 'Components']] = df['Descriptor'].apply(parse_pca_row)

    # B. Parse CV Scores
    def parse_cv(s):
        s = str(s).replace('[', '').replace(']', '')
        return np.array([float(x) for x in s.split()])
    
    cv_arrays = df['CV Scores'].apply(parse_cv)
    df['CV Mean'] = cv_arrays.apply(np.mean)
    df['CV Std'] = cv_arrays.apply(np.std)

    # Sort for correct line plotting
    df = df.sort_values(by=["Method", "Components"])

    # --- 2. Custom Plotting Function ---
    def plot_mean_std_lines(data, **kwargs):
        ax = plt.gca()
        
        # Plot Train (Blue Line)
        ax.plot(
            data['Components'], data['Train Accuracy'], 
            marker='o', markersize=6, linewidth=2, 
            color='#2b7bba', label='Train Accuracy'
        )
        
        # Plot CV Mean (Green Line)
        ax.plot(
            data['Components'], data['CV Mean'], 
            marker='o', markersize=6, linewidth=2, 
            color='#5aa154', label='CV Accuracy (Mean)'
        )
        
        # Plot CV Confidence Band
        ax.fill_between(
            data['Components'], 
            data['CV Mean'] - data['CV Std'], 
            data['CV Mean'] + data['CV Std'], 
            color='#5aa154', alpha=0.2
        )

        # Labels
        for i, row in data.iterrows():
            k = row['Components']
            
            # Label Train
            train_val = row['Train Accuracy']
            ax.text(k, train_val + 0.015, f'{train_val:.3f}', 
                    ha='center', va='bottom', fontsize=8, color='#2b7bba')
            
            # Label CV
            cv_mean = row['CV Mean']
            cv_std = row['CV Std']
            lower_bound = cv_mean - cv_std
            ax.text(k, lower_bound - 0.015, f'{cv_mean:.3f}', 
                    ha='center', va='top', fontsize=8, color='#5aa154')

    # --- 3. FacetGrid Setup ---
    g = sns.FacetGrid(df, col="Method", height=6, aspect=1.2, sharey=True)
    g.map_dataframe(plot_mean_std_lines)

    # --- 4. Styling ---
    unique_n = sorted(df['Components'].unique())

    for ax in g.axes.flat:
        ax.set_xscale('log')
        ax.set_xticks(unique_n)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel("Number of PCA Components (Log Scale)", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        
        # Adjust Y-limits to fit the new labels and bands
        ax.set_ylim(0.1, 0.6)

    # Legend
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    legend_elements = [
        Line2D([0], [0], color='#2b7bba', lw=2, marker='o', label='Train Accuracy'),
        Line2D([0], [0], color='#5aa154', lw=2, marker='o', label='CV Accuracy (Mean)'),
        mpatches.Patch(color='#5aa154', alpha=0.2, label='CV Std Dev Region')
    ]
    plt.legend(handles=legend_elements, loc='upper left', title="Metric", frameon=True)
    
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle("Dimensionality Reduction (PCA): Train vs CV Performance", fontsize=16)
    
    plt.show()


def plot_norm_scale(df):
    """
    Plots results for Normalization and Scaling experiments.
    """
    df = df.copy()

    def parse_cv(s):
        s = str(s).replace('[', '').replace(']', '')
        return np.array([float(x) for x in s.split()])
    
    # Check if string or already list
    if isinstance(df['CV Scores'].iloc[0], str):
        cv_arrays = df['CV Scores'].apply(parse_cv)
    else:
        cv_arrays = df['CV Scores']
        
    df['CV Mean'] = cv_arrays.apply(np.mean)
    df['CV Std'] = cv_arrays.apply(np.std)

    # 2. Advanced Parsing Function
    def parse_descriptor(row):
        parts = row.split('_')
        
        # Logic: 
        # If SIFT_... (3 parts) -> Method=SIFT, Norm=1, Scale=2
        # If Dense_SIFT_... (4 parts) -> Method=Dense SIFT, Norm=2, Scale=3
        
        if parts[0] == "SIFT":
            return pd.Series(["SIFT", parts[1], parts[2]])
        elif parts[0] == "Dense":
            return pd.Series(["Dense SIFT", parts[2], parts[3]])
        else:
            return pd.Series(["Unknown", "None", "None"])

    df[['Method', 'Norm', 'Scale']] = df['Descriptor'].apply(parse_descriptor)

    # 3. Define Categorical Order (To make the plot logical)
    norm_order = ['None', 'L1', 'L2']
    scale_order = ['None', 'Standard', 'MinMax']

    df['Norm'] = pd.Categorical(df['Norm'], categories=norm_order, ordered=True)
    df['Scale'] = pd.Categorical(df['Scale'], categories=scale_order, ordered=True)

    # 4. PLOTTING: FacetGrid Bar Chart (Side-by-Side)
    g = sns.catplot(
        data=df,
        kind="bar",
        x="Norm",
        y="CV Mean",
        hue="Scale",
        col="Method", # Separates SIFT vs Dense SIFT
        palette="viridis",
        height=6,
        aspect=1,
        edgecolor="black",
        linewidth=0.5
    )

    # 5. Styling
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Normalization and Scaling Strategies", fontsize=16)
    g.set_axis_labels("Normalization Strategy", "CV Mean Accuracy")

    # Add numbers on top of bars
    for ax in g.axes.flat:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
        
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        # Set ylim slightly higher than max value to fit labels
        ax.set_ylim(0, 0.45) 

    plt.show()


def plot_norm_scale_v2(df):
    """
    Plots results for Normalization and Scaling experiments (Version 2).
    Cleaned version: No bar labels, no error bars.
    Retains Overfitting Gap annotations.
    """
    df = df.copy()

    # --- 1. Parse CV Scores ---
    def parse_cv(s):
        s = str(s).replace('[', '').replace(']', '')
        return np.array([float(x) for x in s.split()])
    
    # Check if string or already list
    if isinstance(df['CV Scores'].iloc[0], str):
        cv_arrays = df['CV Scores'].apply(parse_cv)
    else:
        cv_arrays = df['CV Scores']
        
    df['CV Mean'] = cv_arrays.apply(np.mean)
    # df['CV Std'] = cv_arrays.apply(np.std) # No longer needed for plot

    # --- 2. Parse Descriptor ---
    def parse_descriptor(row):
        parts = row.split('_')
        if parts[0] == "SIFT":
            # SIFT_Norm_Scale
            return pd.Series(["SIFT", parts[1], parts[2]])
        elif parts[0] == "Dense":
            # Dense_SIFT_Norm_Scale
            return pd.Series(["Dense SIFT", parts[2], parts[3]])
        return pd.Series(["Unknown", "None", "None"])

    df[['Method', 'Norm', 'Scale']] = df['Descriptor'].apply(parse_descriptor)

    # --- 3. Prepare Data ---
    hue_order = ['Train Accuracy', 'CV Mean']
    
    df_melted = df.melt(
        id_vars=['Method', 'Norm', 'Scale'], 
        value_vars=hue_order, 
        var_name='Split', 
        value_name='Accuracy'
    )

    scale_order = ['None', 'Standard', 'MinMax']
    norm_order = ['None', 'L1', 'L2']
    
    scale_order = [x for x in scale_order if x in df['Scale'].unique()]
    norm_order = [x for x in norm_order if x in df['Norm'].unique()]

    # --- 4. Plotting ---
    g = sns.catplot(
        data=df_melted,
        kind="bar",
        x="Norm",
        y="Accuracy",
        hue="Split",
        hue_order=hue_order,
        row="Method",
        col="Scale",
        col_order=scale_order,
        order=norm_order,
        palette="viridis",
        height=4, 
        aspect=0.9,
        legend=True
    )

    # --- 5. Annotations (Overfitting Lines Only) ---
    for row_idx, row_name in enumerate(g.row_names):
        for col_idx, col_name in enumerate(g.col_names):
            ax = g.axes[row_idx, col_idx]
            
            # Draw Overfitting Lines
            if len(ax.containers) >= 2:
                train_bars = ax.containers[0]
                cv_bars = ax.containers[1]
                
                for train_bar, cv_bar in zip(train_bars, cv_bars):
                    if train_bar.get_height() == 0 or cv_bar.get_height() == 0:
                        continue
                        
                    x1 = train_bar.get_x() + train_bar.get_width() / 2
                    y1 = train_bar.get_height()
                    x2 = cv_bar.get_x() + cv_bar.get_width() / 2
                    y2 = cv_bar.get_height()
                    
                    gap = y1 - y2
                    
                    # Draw Line
                    ax.plot([x1, x2], [y1, y2], color='#c44e52', linestyle='--', linewidth=1.5, marker='o', markersize=3)
                    
                    # Gap Label
                    mid_x = (x1 + x2) / 2
                    mid_y = max(y1, y2) + 0.05 
                    
                    ax.text(
                        mid_x, mid_y, 
                        f"+{gap:.3f}", 
                        ha='center', va='bottom', 
                        fontsize=8, color='#c44e52', fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
                    )
            
            ax.set_ylim(0, 0.7) # Adjust y-limit as needed
            ax.grid(axis='y', linestyle='--', alpha=0.5)

    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Normalization and Scaling: Train vs CV Performance", fontsize=16)
    g.set_axis_labels("Normalization", "Accuracy")
    
    plt.show()


def plot_spatial_pyramid(df):
    """
    Plots results for Spatial Pyramid experiments using CV Scores.
    Removes Error Bars but keeps Overfitting Gap annotations.
    """
    df_plot = df.copy()
    
    # --- 1. Data Parsing ---
    def parse_pyramid_row(row):
        # Extract Level number: "SIFT (L1)" -> 1
        level_match = re.search(r'\(L(\d+)\)', row)
        level_val = int(level_match.group(1)) if level_match else 0
        
        # Extract Method Name: "SIFT (L1)" -> "SIFT"
        name = re.sub(r'\(.*?\)', '', row).strip()
        return pd.Series([name, level_val])

    df_plot[['Method', 'Levels']] = df_plot['Descriptor'].apply(parse_pyramid_row)
    
    # Parse CV Scores
    def parse_cv(s):
        s = str(s).replace('[', '').replace(']', '')
        return np.array([float(x) for x in s.split()])
    
    cv_arrays = df_plot['CV Scores'].apply(parse_cv)
    df_plot['CV Mean'] = cv_arrays.apply(np.mean)
    # df_plot['CV Std'] = cv_arrays.apply(np.std) # No longer needed for plot height
    
    # Melt for plotting
    hue_order = ["Train Accuracy", "CV Mean"]
    df_melted = df_plot.melt(
        id_vars=["Method", "Levels"], 
        value_vars=hue_order, 
        var_name="Split", 
        value_name="Accuracy"
    )
    df_melted = df_melted.sort_values(by=["Method", "Levels"])
    
    # --- 2. Plotting ---
    g = sns.catplot(
        data=df_melted,
        kind="bar",
        x="Levels",
        y="Accuracy",
        hue="Split",
        col="Method",
        palette="viridis",
        hue_order=hue_order,
        height=6,
        aspect=1,
        sharey=False 
    )
    
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Spatial Pyramid Levels: Train vs CV Performance", fontsize=16)
    g.set_axis_labels("Pyramid Level", "Accuracy")
    
    # Iterate over axes to add custom annotations
    methods = sorted(df_plot['Method'].unique())
    
    for ax, method in zip(g.axes.flat, methods):
        # Get data for this subplot to ensure alignment (though bar_label handles most)
        subset = df_plot[df_plot['Method'] == method].sort_values('Levels')
        
        # A. Label Train Bars (Container 0)
        ax.bar_label(ax.containers[0], fmt='%.3f', padding=3, fontsize=9)
        
        # B. Label CV Bars (Container 1) - Manual Labels (No Error Bars)
        cv_container = ax.containers[1]
        means = subset['CV Mean'].values
        
        for bar, mean in zip(cv_container, means):
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            
            # REMOVED ax.errorbar(...)
            
            # Adjusted label position to be just above the bar
            label_y = y + 0.005 
            ax.text(x, label_y, f'{mean:.3f}', ha='center', va='bottom', fontsize=9, color='black')

        # C. Overfitting Lines (Train vs CV Mean)
        train_bars = ax.containers[0]
        cv_bars = ax.containers[1]
        
        for train_bar, cv_bar in zip(train_bars, cv_bars):
             x1 = train_bar.get_x() + train_bar.get_width() / 2
             y1 = train_bar.get_height()
             x2 = cv_bar.get_x() + cv_bar.get_width() / 2
             y2 = cv_bar.get_height()
             
             gap = y1 - y2
             
             # Draw dashed line connecting top of Train bar to top of CV bar
             ax.plot([x1, x2], [y1, y2], color='#c44e52', linestyle='--', linewidth=1.5, marker='o', markersize=4)
             
             # Text Box for the Gap
             mid_x = x2 
             mid_y = (y1 + y2) / 2
             
             ax.text(
                mid_x, mid_y, 
                f"+{gap:.3f}", 
                ha='center', va='bottom', 
                fontsize=9, 
                color='#c44e52', 
                fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
            )
            
        ax.set_ylim(0, 0.7) 
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.show()


def plot_logistic(df):
    """
    Plots results for Logistic Regression experiments using CV Scores.
    Compares Train Accuracy vs CV Accuracy (Mean ± Std).
    """
    df = df.copy()

    # --- 1. Data Parsing ---
    # Separate "Method" from "C" value
    def parse_logistic_row(row):
        # Extract C value: "LogisticRegression (C=0.01)" -> 0.01
        c_match = re.search(r'C=([0-9.]+)', row)
        c_val = float(c_match.group(1)) if c_match else 0.0
        
        name = row.split('(')[0].strip()
        return pd.Series([name, c_val])

    df[['Method', 'C']] = df['Descriptor'].apply(parse_logistic_row)

    # Parse CV Scores
    def parse_cv(s):
        s = str(s).replace('[', '').replace(']', '')
        return np.array([float(x) for x in s.split()])
    
    cv_arrays = df['CV Scores'].apply(parse_cv)
    df['CV Mean'] = cv_arrays.apply(np.mean)
    df['CV Std'] = cv_arrays.apply(np.std)

    # Sort by C for correct line plotting
    df_sorted = df.sort_values(by="C")

    # --- 2. Plotting ---
    plt.figure(figsize=(10, 6))
    
    # Plot Train Accuracy (Blue Line)
    plt.plot(
        df_sorted['C'], df_sorted['Train Accuracy'], 
        marker='o', markersize=8, linewidth=2.5, 
        color='#2b7bba', label='Train Accuracy'
    )
    
    # Plot CV Mean (Green Line)
    plt.plot(
        df_sorted['C'], df_sorted['CV Mean'], 
        marker='o', markersize=8, linewidth=2.5, 
        color='#5aa154', label='CV Accuracy (Mean)'
    )
    
    # Plot CV Confidence Band (Shaded Region)
    plt.fill_between(
        df_sorted['C'], 
        df_sorted['CV Mean'] - df_sorted['CV Std'], 
        df_sorted['CV Mean'] + df_sorted['CV Std'], 
        color='#5aa154', alpha=0.2, label='CV Std Dev Region'
    )
    
    # --- 3. Formatting ---
    ax = plt.gca()
    ax.set_xscale('log')
    
    # Set ticks to specific C values (0.01, 0.1, 1, 10)
    unique_c = sorted(df['C'].unique())
    ax.set_xticks(unique_c)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlabel("Regularization Parameter C (log scale)", fontsize=12)
    plt.title("Logistic Regression: Regularization C (Train vs CV)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # --- 4. Add Labels ---
    # Train Labels: Above the point
    for x, y in zip(df_sorted['C'], df_sorted['Train Accuracy']):
        plt.text(x, y + 0.015, f'{y:.3f}', ha='center', va='bottom', fontsize=9, color='#2b7bba', fontweight='bold')
        
    # CV Labels: Below the error band
    for x, mean, std in zip(df_sorted['C'], df_sorted['CV Mean'], df_sorted['CV Std']):
        # Position label below the bottom of the error band
        label_y = mean - std - 0.015
        plt.text(x, label_y, f'{mean:.3f}', ha='center', va='top', fontsize=9, color='#5aa154', fontweight='bold')

    plt.legend(title="Metric", loc='upper left')
    
    # Adjust Y-limits to fit labels comfortably
    ymax = max(df_sorted['Train Accuracy'].max(), (df_sorted['CV Mean'] + df_sorted['CV Std']).max())
    ymin = min(df_sorted['Train Accuracy'].min(), (df_sorted['CV Mean'] - df_sorted['CV Std']).min())
    plt.ylim(ymin - 0.05, ymax + 0.05)

    plt.tight_layout()
    plt.show()


def plot_svm(df):
    """
    Plots results for SVM experiments using CV Scores.
    Compares Train Accuracy vs CV Accuracy (Mean ± Std) across different Kernels.
    """
    df = df.copy()
    
    # --- 1. Data Parsing ---
    def parse_svm_row(row):
        # Extract C value
        c_match = re.search(r'C=([0-9.]+)', row)
        c_val = float(c_match.group(1)) if c_match else 0.0
        
        # Extract Method Name (Kernel)
        name = row.split('(')[0].strip()
        return pd.Series([name, c_val])

    df[['Method', 'C']] = df['Descriptor'].apply(parse_svm_row)

    # Parse CV Scores
    def parse_cv(s):
        s = str(s).replace('[', '').replace(']', '')
        return np.array([float(x) for x in s.split()])
    
    cv_arrays = df['CV Scores'].apply(parse_cv)
    df['CV Mean'] = cv_arrays.apply(np.mean)
    df['CV Std'] = cv_arrays.apply(np.std)

    # Sort for correct line plotting
    df = df.sort_values(by=["Method", "C"])

    # --- 2. Custom Plotting Function ---
    def plot_mean_std_lines(data, **kwargs):
        ax = plt.gca()
        
        # Plot Train (Blue Line)
        ax.plot(
            data['C'], data['Train Accuracy'], 
            marker='o', markersize=6, linewidth=2, 
            color='#2b7bba', label='Train Accuracy'
        )
        
        # Plot CV Mean (Green Line)
        ax.plot(
            data['C'], data['CV Mean'], 
            marker='o', markersize=6, linewidth=2, 
            color='#5aa154', label='CV Accuracy (Mean)'
        )
        
        # Plot CV Confidence Band
        ax.fill_between(
            data['C'], 
            data['CV Mean'] - data['CV Std'], 
            data['CV Mean'] + data['CV Std'], 
            color='#5aa154', alpha=0.2
        )

        # Labels
        for i, row in data.iterrows():
            c = row['C']
            
            # Label Train
            train_val = row['Train Accuracy']
            ax.text(c, train_val + 0.03, f'{train_val:.3f}', 
                    ha='center', va='bottom', fontsize=8, color='#2b7bba')
            
            # Label CV
            cv_mean = row['CV Mean']
            cv_std = row['CV Std']
            lower_bound = cv_mean - cv_std
            ax.text(c, lower_bound - 0.03, f'{cv_mean:.3f}', 
                    ha='center', va='top', fontsize=8, color='#5aa154')

    # --- 3. FacetGrid Setup ---
    g = sns.FacetGrid(df, col="Method", height=6, aspect=1, sharey=True)
    g.map_dataframe(plot_mean_std_lines)

    # --- 4. Styling ---
    unique_c = sorted(df['C'].unique())

    for ax in g.axes.flat:
        ax.set_xscale('log')
        ax.set_xticks(unique_c)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel("Regularization C (log scale)", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        
        # Set Y-limits to handle the full range (0 to 1+) and label padding
        ax.set_ylim(-0.1, 1.15) 

    # Legend (Manually constructed for FacetGrid)
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    legend_elements = [
        Line2D([0], [0], color='#2b7bba', lw=2, marker='o', label='Train Accuracy'),
        Line2D([0], [0], color='#5aa154', lw=2, marker='o', label='CV Accuracy (Mean)'),
        mpatches.Patch(color='#5aa154', alpha=0.2, label='CV Std Dev Region')
    ]
    plt.legend(handles=legend_elements, loc='upper left', title="Metric", frameon=True)
    
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle("SVM Kernel Comparison: Regularization C (Train vs CV)", fontsize=16)
    
    plt.show()


def plot_svm_rbf_gamma(df):
    """
    Plots results for SVM RBF Gamma experiments using CV Scores.
    Includes Error Bars and Overfitting Gap annotations.
    """
    df = df.copy()

    # --- 1. Data Parsing ---
    def parse_gamma_row(row):
        match = re.search(r'Gamma=([A-Za-z0-9.]+)', row)
        return match.group(1) if match else "unknown"

    df['Gamma'] = df['Descriptor'].apply(parse_gamma_row)

    # Parse CV Scores
    def parse_cv(s):
        s = str(s).replace('[', '').replace(']', '')
        return np.array([float(x) for x in s.split()])
    
    cv_arrays = df['CV Scores'].apply(parse_cv)
    df['CV Mean'] = cv_arrays.apply(np.mean)
    df['CV Std'] = cv_arrays.apply(np.std)

    # Prepare for Plotting
    hue_order = ["Train Accuracy", "CV Mean"]
    df_melted = df.melt(
        id_vars=["Gamma"], 
        value_vars=hue_order, 
        var_name="Split", 
        value_name="Accuracy"
    )

    # Define logical order for Gamma
    custom_order = ["scale", "auto", "0.1", "1", "10", "100"]
    present_order = [x for x in custom_order if x in df['Gamma'].unique()]

    # --- 2. Plotting ---
    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=df_melted,
        x="Gamma",
        y="Accuracy",
        hue="Split",
        order=present_order,
        palette="viridis",
        hue_order=hue_order
    )

    # --- 3. Labels and Annotations ---
    
    # A. Train Accuracy Labels (Container 0)
    ax.bar_label(ax.containers[0], fmt='%.3f', padding=3, fontsize=9)

    # B. CV Mean Labels + Error Bars (Container 1)
    cv_container = ax.containers[1]
    
    # We iterate through the bars and the dataframe to find the matching Std Dev
    for i, gamma in enumerate(present_order):
        bar = cv_container[i]
        
        # Get data for this Gamma
        row = df[df['Gamma'] == gamma].iloc[0]
        mean = row['CV Mean']
        std = row['CV Std']
        
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        
        # Error Bar
        ax.errorbar(x, y, yerr=std, fmt='none', c='black', capsize=4)
        
        # Label (Above error bar)
        label_y = y + std + 0.015
        ax.text(x, label_y, f'{mean:.3f}', ha='center', va='bottom', fontsize=9, color='black')

    # C. Overfitting Lines (Train vs CV Gap)
    train_bars = ax.containers[0]
    cv_bars = ax.containers[1]
    
    for train_bar, cv_bar in zip(train_bars, cv_bars):
        x1 = train_bar.get_x() + train_bar.get_width() / 2
        y1 = train_bar.get_height()
        x2 = cv_bar.get_x() + cv_bar.get_width() / 2
        y2 = cv_bar.get_height()
        
        gap = y1 - y2
        
        # Draw dashed line
        ax.plot([x1, x2], [y1, y2], color='#c44e52', linestyle='--', marker='o', linewidth=1.5, markersize=4)
        
        # Text Box for Gap
        mid_x = x2 # Align closer to CV bar
        mid_y = (y1 + y2) / 2
        
        ax.text(
            mid_x, mid_y, 
            f"+{gap:.3f}", 
            color='#c44e52', 
            ha='center', va='bottom', 
            fontsize=9, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
        )

    plt.title("SVM-RBF: Gamma Parameter (Train vs CV)", fontsize=14)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Gamma Parameter", fontsize=12)
    plt.ylim(0, 1.25) # Increase headroom for annotations
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Dataset Split", loc='upper left')

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# Explanatory visualizations
# -------------------------------------------------------------------------


def visualize_dense_sift_grid(image_path, step_sizes, scale_factors):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # 1. Load and Crop Image
    # We crop the image to a smaller region (e.g., 200x200) so we can actually SEE the circles.
    # If we use the full image, Step=4 looks like a solid block of color.
    img_orig = cv2.imread(image_path)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    
    # Crop a central patch for better visibility
    h, w = img_orig.shape[:2]
    crop_size = 192
    cy, cx = h // 2, w // 2
    img = img_orig[cy-crop_size//2 : cy+crop_size//2, cx-crop_size//2 : cx+crop_size//2]
    h, w = img.shape[:2]

    # 2. Setup Plot Grid
    n_rows = len(step_sizes)
    n_cols = len(scale_factors)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    plt.suptitle(f"Dense SIFT Visualization", fontsize=16, y=1)

    # 3. Iterate Parameters
    for i, step in enumerate(step_sizes):
        for j, factor in enumerate(scale_factors):
            ax = axes[i, j] if n_rows > 1 else axes[j]
            
            # Calculate Scale based on your formula
            # Ensure scale is at least 1 pixel
            scale = max(1, int(step / factor))
            
            # Generate Keypoints (Manual Grid)
            keypoints = []
            for y in range(step // 2, h, step):
                for x in range(step // 2, w, step):
                    # OpenCV KeyPoint takes (x, y, size)
                    # size is the diameter of the SIFT neighborhood
                    keypoints.append(cv2.KeyPoint(float(x), float(y), float(scale)))

            # Draw Keypoints
            # DRAW_RICH_KEYPOINTS flag draws the circle representing the scale
            img_with_kps = cv2.drawKeypoints(img, keypoints, None, 
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                           color=(-1, -1, -1)) # Green circles

            # Plot
            ax.imshow(img_with_kps)
            ax.set_title(f"Step: {step}px | Factor: {factor}\nScale (Diam): {scale}px")
            ax.axis('off')

    plt.tight_layout()
    plt.show()



def visualize_spatial_pyramid(image_path: str, bovw: BOVW):
    """
    Visualizes the Spatial Pyramid grids and the resulting histograms per level.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    # 1. Load Image
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]

    # 2. Extract Features (Using your class method)
    # Your _extract_features returns (keypoints, descriptors)
    kps, descs = bovw._extract_features(img_np)
    
    if descs is None or len(descs) == 0:
        print("No features found in image.")
        return

    # Convert KeyPoint objects to (x, y) coordinates for plotting
    kp_positions = np.array([kp.pt for kp in kps])

    # 3. Check if Codebook is fitted (needed to show histograms)
    # If not fitted, we fit it temporarily on this single image just for visualization
    try:
        bovw.codebook_algo.predict(descs[0:1])
        is_fitted = True
    except:
        print("Warning: Codebook not fitted. Fitting temporarily on this image for visualization.")
        bovw._update_fit_codebook([descs])
        is_fitted = True
    
    # Predict visual words for the image
    visual_words = bovw.codebook_algo.predict(descs)

    # 4. Create Plots (Rows = Levels)
    levels = bovw.levels
    fig, axes = plt.subplots(len(levels), 2, figsize=(15, 5 * len(levels)))
    if len(levels) == 1: axes = [axes] # Handle single level case

    print(f"--- Visualizing {len(levels)} Levels ---")

    for idx, level in enumerate(levels):
        # Setup Axes
        ax_img = axes[idx][0] if len(levels) > 1 else axes[0]
        ax_hist = axes[idx][1] if len(levels) > 1 else axes[1]
        
        # Show Image
        ax_img.imshow(img_np)
        ax_img.set_title(f"Level {idx}: Grid {level}x{level}")
        
        step_x = w / level
        step_y = h / level

        # --- DRAW GRID & CALCULATE HISTOGRAM ---
        
        # We will collect histograms for this level to plot them
        level_histograms = []
        cell_labels = []
        
        cell_id = 0
        
        # Iterate Rows and Cols (Same logic as your _compute_spatial_pyramid_descriptor)
        for r in range(level):
            for c in range(level):
                # Coordinates
                x_min, x_max = c * step_x, (c+1) * step_x
                y_min, y_max = r * step_y, (r+1) * step_y

                # Draw Lines
                rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                                         linewidth=2, edgecolor='red', facecolor='none', alpha=0.7)
                ax_img.add_patch(rect)
                
                # Filter points in this cell
                in_x = (kp_positions[:, 0] >= x_min) & (kp_positions[:, 0] < x_max)
                in_y = (kp_positions[:, 1] >= y_min) & (kp_positions[:, 1] < y_max)
                in_cell_indices = np.where(in_x & in_y)[0]
                
                # Plot points (Optional: only plot a subset to avoid clutter)
                if len(in_cell_indices) > 0:
                    ax_img.scatter(kp_positions[in_cell_indices, 0], kp_positions[in_cell_indices, 1], 
                                   s=5, alpha=0.5, label=f'Cell {cell_id}')

                # Add Cell ID Text
                ax_img.text(x_min + 5, y_min + 20, f"Cell {cell_id}", 
                            color='white', fontweight='bold', bbox=dict(facecolor='red', alpha=0.5))

                # Compute Histogram for this specific cell
                hist = np.zeros(bovw.codebook_size)
                if len(in_cell_indices) > 0:
                    cell_words = visual_words[in_cell_indices]
                    for w_idx in cell_words:
                        hist[w_idx] += 1
                
                # Normalize local histogram for visualization clarity
                if np.sum(hist) > 0: hist = hist / np.sum(hist)
                
                level_histograms.append(hist)
                cell_labels.append(f"Cell {cell_id}")
                cell_id += 1

        ax_img.axis('off')

        # --- PLOT HISTOGRAM FOR THIS LEVEL ---
        # We stack the histograms of all cells in this level side-by-side
        # This shows how the feature vector is constructed (concatenation)
        
        combined_hist = np.concatenate(level_histograms)
        ax_hist.bar(range(len(combined_hist)), combined_hist, width=1.0)
        
        # Draw vertical lines separating cells in the histogram
        for i in range(1, len(level_histograms)):
            ax_hist.axvline(x=i * bovw.codebook_size, color='red', linestyle='--')
            
        ax_hist.set_title(f"Concatenated Histogram for Level {level}x{level}")
        ax_hist.set_xlabel("Feature Index (concatenated cells)")
        ax_hist.set_ylabel("Normalized Frequency")
        
        # Add text to indicate which section belongs to which cell
        for i in range(len(level_histograms)):
            mid_point = (i * bovw.codebook_size) + (bovw.codebook_size / 2)
            ax_hist.text(mid_point, max(combined_hist)*0.9, f"Cell {i}", 
                         ha='center', color='red', fontweight='bold')

    plt.tight_layout()
    plt.show()