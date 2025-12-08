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
    Internal helper for standard grouped bar plots (Train vs Test).
    """
    plt.figure(figsize=(10, 6))
    
    # Melt for plotting
    df_melted = df.melt(
        id_vars=[x_col], 
        value_vars=["Train Accuracy", "Test Accuracy"], 
        var_name=hue_col, 
        value_name="Accuracy"
    )
    
    ax = sns.barplot(
        x=x_col,
        y="Accuracy",
        hue=hue_col,
        data=df_melted,
        palette="viridis"
    )
    
    # Add labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)
    
    # Overfitting lines logic (if applicable, assuming ordered bars match)
    # This logic assumes simple pairing 
    containers = ax.containers
    if len(containers) >= 2:
        # container 0 is Train (usually), container 1 is Test
        train_bars = containers[0]
        test_bars = containers[1]
        
        for train_bar, test_bar in zip(train_bars, test_bars):
            # Calculate coordinates
            x1 = train_bar.get_x() + train_bar.get_width() / 2
            y1 = train_bar.get_height()
            x2 = test_bar.get_x() + test_bar.get_width() / 2
            y2 = test_bar.get_height()
            
            gap = y1 - y2
            mid_x = (x1 + x2) / 2
            mid_y = max(y1, y2) + 0.02
            
            # Plot line
            plt.plot([x1, x2], [y1, y2], color='#c44e52', linestyle='--', marker='o', linewidth=1.5, markersize=4)
            # Add text
            plt.text(mid_x, mid_y, f"+{gap:.3f}", color='#c44e52', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.title(title, fontsize=16)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(title="Split")
    if ylim:
        plt.ylim(ylim)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_local_descriptors(df):
    """
    Plots results for Local Descriptors (SIFT, ORB, AKAZE).
    """
    _plot_grouped_bar(df, x_col="Descriptor", hue_col="Split", title="Local Descriptors Performance", ylim=(0, 0.35))

def plot_sift_nfeatures(df):
    """
    Plots results for SIFT NFeatures as a line plot with log scale.
    """
    # 2. Extract Integers for the Continuous X-Axis
    # We use regex to find the numbers inside the string and convert to int
    df = df.copy()
    df['n_features'] = df['Descriptor'].str.extract(r'(\d+)').astype(int)

    # 3. Melt Data
    df_melted = df.melt(
        id_vars=["n_features"], 
        value_vars=["Train Accuracy", "Test Accuracy"], 
        var_name="Split", 
        value_name="Accuracy"
    )

    plt.figure(figsize=(12, 6))

    # 4. Plot using the numeric 'n_features' column
    ax = sns.lineplot(
        x="n_features", 
        y="Accuracy", 
        hue="Split", 
        data=df_melted, 
        palette="viridis",
        marker="o",
        markersize=8
    )

    # 5. Intelligent Label Placement
    # We iterate through the dataframe rows to determine label position based on "Split"
    for index, row in df_melted.iterrows():
        x = row['n_features']
        y = row['Accuracy']
        split = row['Split']

            
        plt.text(
            x, 
            y + 0.002, 
            f'{y:.3f}', 
            ha='center', 
            va='bottom', 
            fontsize=9,
            color='black' # Optional: match text color to line color if desired
        )

    # Formatting
    plt.title("SIFT: Number of Features")
    plt.ylim(0.2, 0.35) # Adjusted to give space for labels
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Features (Integer)")

    # Set X-Axis to Log Scale 
    # (Since 50 vs 10000 is a huge gap, a linear scale squashes the start. 
    # Log scale spreads them out nicely like your original image).
    plt.xscale('log') 

    # Customize X-ticks so they show the actual integers, not scientific notation
    tick_vals = df['n_features'].unique()
    # Ensure tick_vals are sorted for display
    tick_vals.sort()
    plt.xticks(tick_vals, tick_vals)

    plt.legend(title="Dataset Split", loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_dense_sift_step_size_scale(df):
    """
    Plots results for Dense SIFT Step Size & Scale.
    Parses 'Descriptor' to get 'Step Size' and 'Scale Factor'.
    """
    df = df.copy()
    
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
        y="Test Accuracy",
        hue="Scale Factor",
        palette="viridis"
    )

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

    plt.title("Dense SIFT: Step Size and Scale")
    plt.ylim(0, 0.6) # Adjust this limit based on your max results
    plt.ylabel("Test Accuracy")
    plt.xlabel("Step Size (px)")
    plt.legend(title="Scale Factor", loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.show()

def plot_codebook_size(df):
    """
    Plots results for Codebook Size experiments using FacetGrid.
    Sparsely labels codebook sizes on log scale.
    """
    df = df.copy()
    
    # 2. Parsing: Separate "Method" from "Codebook_Size"
    def parse_row(row):
        # Extract number
        k_match = re.search(r'k=(\d+)', str(row))
        k = int(k_match.group(1)) if k_match else 0
        # Extract name (everything before parenthesis or underscore k if simpler format)
        # User prompt regex logic: name = row.split('(')[0].strip()
        # Adapting to robustly handle typical row strings
        if '(' in str(row):
             name = str(row).split('(')[0].strip()
        else:
             # Fallback if no parens, try to split by _k
             name = str(row).split('_k')[0].strip()
        return pd.Series([name, k])

    df[['Method', 'Codebook_Size']] = df['Descriptor'].apply(parse_row)

    # 3. Melt (Train vs Test)
    df_melted = df.melt(
        id_vars=["Method", "Codebook_Size"], 
        value_vars=["Train Accuracy", "Test Accuracy"], 
        var_name="Split", 
        value_name="Accuracy"
    )

    # Sort for clean plotting
    df_melted = df_melted.sort_values(by=["Method", "Codebook_Size"])

    # 4. PLOTTING with FacetGrid
    g = sns.FacetGrid(df_melted, col="Method", height=6, aspect=1.1, sharey=True)

    # Map the lineplot
    g.map_dataframe(
        sns.lineplot, 
        x="Codebook_Size", 
        y="Accuracy", 
        hue="Split", 
        palette="viridis", 
        marker="o",
        linewidth=2.5,
        markersize=7
    )

    # 5. Apply Custom Styling
    unique_k = sorted(df['Codebook_Size'].unique())

    for ax in g.axes.flat:
        ax.set_ylim(0, 1.15) # Keep original robust limits or user's requested 0.15-0.6 depending on context? User asked for 0.15, 0.6. I will respect user request but maybe widen if it cuts off data. Let's stick to user request.
        ax.set_ylim(0.15, 0.6)
        
        ax.set_xscale('log')
        ax.set_xticks(unique_k)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Codebook Size k (log scale)")

    # 6. INTELLIGENT TEXT LABELING
    # Iterate over methods (subplots)
    methods = sorted(df_melted['Method'].unique())

    # Zip the axes with the methods to ensure we plot on the right graph
    for ax, method in zip(g.axes.flat, methods):
        # Get only the data for this specific method
        subset = df_melted[df_melted['Method'] == method]
        
        for _, row in subset.iterrows():
            # --- LOGIC START ---
            if "Test" in row['Split']:
                # If Test: Label BELOW the point
                xy_text = (row['Codebook_Size'], row['Accuracy'] - 0.012)
                valign = 'top' 
            else:
                # If Train: Label ABOVE the point
                xy_text = (row['Codebook_Size'], row['Accuracy'] + 0.012)
                valign = 'bottom'
            # --- LOGIC END ---

            ax.text(
                xy_text[0], 
                xy_text[1], 
                f"{row['Accuracy']:.3f}", 
                ha='center', 
                va=valign, 
                fontsize=9, 
                color='black',
                fontweight='normal'
            )

    # Add Legend and Title
    g.add_legend(title="Dataset Split")
    plt.subplots_adjust(top=0.85) 
    g.fig.suptitle("Codebook Size Comparison", fontsize=16)

    plt.show()

def plot_dim_reduction(df):
    """
    Plots results for Dimensionality Reduction (PCA) experiments.
    """
    df = df.copy()
    
    # 2. Parsing Logic
    def parse_pca_row(row):
        # Extract number
        n_match = re.search(r'(\d+)', str(row))
        n_val = int(n_match.group(1)) if n_match else 0
        # Extract name
        name = re.sub(r'\(.*?\)', '', str(row)).strip() 
        name = re.sub(r'\d+', '', name).strip()
        return pd.Series([name, n_val])

    df[['Method', 'Components']] = df['Descriptor'].apply(parse_pca_row)

    # 3. Melt
    df_melted = df.melt(
        id_vars=["Method", "Components"], 
        value_vars=["Train Accuracy", "Test Accuracy"], 
        var_name="Split", 
        value_name="Accuracy"
    )
    df_melted = df_melted.sort_values(by=["Method", "Components"])

    # 4. Plotting
    g = sns.FacetGrid(df_melted, col="Method", height=6, aspect=1.2, sharey=True)

    g.map_dataframe(
        sns.lineplot, 
        x="Components", 
        y="Accuracy", 
        hue="Split", 
        palette="viridis", 
        marker="o",
        linewidth=2.5,
        markersize=8
    )

    # 5. Styling with Log Scale
    unique_n = sorted(df['Components'].unique())

    for ax in g.axes.flat:
        ax.set_ylim(0.15, 0.5) # Give a little headroom for text
        
        # --- LOG SCALE SETUP ---
        ax.set_xscale('log')
        ax.set_xticks(unique_n)
        # ScalarFormatter prevents "10^1" notation and forces "16, 32" etc.
        ax.get_xaxis().set_major_formatter(ScalarFormatter()) 
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Number of PCA Components (Log Scale)")

    # 6. Intelligent Labeling (Test Bottom / Train Top)
    methods = df_melted['Method'].unique()

    # Zip axes with methods to label correctly
    for ax, method in zip(g.axes.flat, methods):
        subset = df_melted[df_melted['Method'] == method]
        
        for _, row in subset.iterrows():
            # --- LOGIC START ---
            if "Test" in row['Split']:
                # Test: Below the dot
                offset = -0.015
                valign = 'top'
            else:
                # Train: Above the dot
                offset = +0.015
                valign = 'bottom'
            # --- LOGIC END ---

            ax.text(
                row['Components'], 
                row['Accuracy'] + offset, 
                f"{row['Accuracy']:.3f}", 
                ha='center', 
                va=valign, 
                fontsize=9, 
                color='black'
            )

    g.add_legend(title="Dataset Split")
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle("Dimensionality Reduction (PCA)", fontsize=16)

    plt.show()

def plot_norm_scale(df):
    """
    Plots results for Normalization and Scaling experiments.
    """
    df = df.copy()

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
        y="Test Accuracy",
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
    g.set_axis_labels("Normalization Strategy", "Test Accuracy")

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
    Includes overfitting lines and facet grid layout.
    """
    df = df.copy()

    # 2. Parse Data
    def parse_descriptor(row):
        parts = row.split('_')
        if parts[0] == "SIFT":
            return pd.Series(["SIFT", parts[1], parts[2]])
        elif parts[0] == "Dense":
            return pd.Series(["Dense SIFT", parts[2], parts[3]])
        return pd.Series(["Unknown", "None", "None"])

    df[['Method', 'Norm', 'Scale']] = df['Descriptor'].apply(parse_descriptor)

    # 3. Melt
    df_melted = df.melt(
        id_vars=['Method', 'Norm', 'Scale'], 
        value_vars=['Train Accuracy', 'Test Accuracy'], 
        var_name='Split', 
        value_name='Accuracy'
    )

    # 4. Define Order
    scale_order = ['None', 'Standard', 'MinMax']
    norm_order = ['None', 'L1', 'L2']
    split_order = ['Train Accuracy', 'Test Accuracy'] 

    # 5. PLOT: Side-by-Side Bar Chart (Catplot)
    g = sns.catplot(
        data=df_melted,
        kind="bar",
        x="Norm",
        y="Accuracy",
        hue="Split",
        hue_order=split_order,
        row="Method",
        col="Scale",
        col_order=scale_order,
        order=norm_order,
        palette="viridis",
        height=4, 
        aspect=0.9,
        legend=True
    )

    # 6. ADD OVERFITTING LINES
    for ax in g.axes.flat:
        # --- SET Y-LIMIT TO 0.6 HERE ---
        ax.set_ylim(0, 0.6)
        
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        if len(ax.containers) >= 2:
            train_bars = ax.containers[0] 
            test_bars = ax.containers[1]  
            
            for train_bar, test_bar in zip(train_bars, test_bars):
                x1 = train_bar.get_x() + train_bar.get_width() / 2
                y1 = train_bar.get_height()
                x2 = test_bar.get_x() + test_bar.get_width() / 2
                y2 = test_bar.get_height()
                gap = y1 - y2
                
                # Draw line
                ax.plot([x1, x2], [y1, y2], color='#c44e52', linestyle='--', linewidth=1.5, marker='o', markersize=3)
                
                # Add label
                mid_x = (x1 + x2) / 2
                mid_y = max(y1, y2) + 0.02 
                
                ax.text(
                    mid_x, 
                    mid_y, 
                    f"+{gap:.3f}", 
                    ha='center', 
                    va='bottom', 
                    fontsize=8, 
                    color='#c44e52', 
                    fontweight='bold'
                )

    # 7. Styling
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Normalization and Scaling Strategies", fontsize=16)
    g.set_axis_labels("Normalization", "Accuracy")

    plt.show()

def plot_spatial_pyramid(df):
    """
    Plots results for Spatial Pyramid experiments.
    """
    df_plot = df.copy()
    
    # --- Data Parsing ---
    def parse_pyramid_row(row):
        # Extract Level number
        level_match = re.search(r'\(L(\d+)\)', row)
        level_val = int(level_match.group(1)) if level_match else 0
        # Extract Method Name
        name = re.sub(r'\(.*?\)', '', row).strip()
        return pd.Series([name, level_val])

    df_plot[['Method', 'Levels']] = df_plot['Descriptor'].apply(parse_pyramid_row)
    
    df_melted = df_plot.melt(
        id_vars=["Method", "Levels"], 
        value_vars=["Train Accuracy", "Test Accuracy"], 
        var_name="Split", 
        value_name="Accuracy"
    )
    df_melted = df_melted.sort_values(by=["Method", "Levels"])
    
    # --- Plotting ---
    g = sns.catplot(
        data=df_melted,
        kind="bar",
        x="Levels",
        y="Accuracy",
        hue="Split",
        col="Method",
        palette="viridis",
        height=6,
        aspect=1,
        sharey=False, # Changed to False so each plot fits its own data height better
        legend=True
    )
    
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Spatial Pyramid Levels", fontsize=16)
    g.set_axis_labels("Pyramid Level", "Accuracy")
        
    for ax in g.axes.flat:
        ax.set_ylim(0, 0.6) 
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Add black labels to bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
            
        # Add Overfitting lines and Red labels
        if len(ax.containers) >= 2:
            train_bars = ax.containers[0]
            test_bars = ax.containers[1]
            
            for train_bar, test_bar in zip(train_bars, test_bars):
                x1 = train_bar.get_x() + train_bar.get_width() / 2
                y1 = train_bar.get_height()
                x2 = test_bar.get_x() + test_bar.get_width() / 2
                y2 = test_bar.get_height()
                
                gap = y1 - y2
                
                # Draw the dotted line
                ax.plot([x1, x2], [y1, y2], color='#c44e52', linestyle='--', linewidth=1.5, marker='o', markersize=4)
                
                # Calculate text position
                mid_x = x2
                mid_y = (y1 + y2)/2
                
                ax.text(
                    mid_x, mid_y, 
                    f"+{gap:.3f}", 
                    ha='center', va='bottom', 
                    fontsize=9, 
                    color='#c44e52', 
                    fontweight='bold',
                    # Add a white box background to prevent visual clutter
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
                )

    plt.show()

def plot_logistic(df):
    """
    Plots results for Logistic Regression experiments.
    """
    df = df.copy()

    # Parsing: Separate "Method" from "C" value
    # Example: "LogisticRegression (C=0.01)" -> Method="LogisticRegression", C=0.01
    def parse_logistic_row(row):
        c_match = re.search(r'C=([0-9.]+)', row)
        c_val = float(c_match.group(1)) if c_match else 0.0
        name = row.split('(')[0].strip()
        return pd.Series([name, c_val])

    df[['Method', 'C']] = df['Descriptor'].apply(parse_logistic_row)

    df_melted = df.melt(
        id_vars=["Method", "C"], 
        value_vars=["Train Accuracy", "Test Accuracy"], 
        var_name="Split", 
        value_name="Accuracy"
    )

    df_melted = df_melted.sort_values(by=["C"])

    plt.figure(figsize=(10, 6))
    
    ax = sns.lineplot(
        data=df_melted,
        x="C",
        y="Accuracy",
        hue="Split",
        palette="viridis",
        marker="o",
        linewidth=2.5
    )

    ax.set_xscale('log')
    
    unique_c = sorted(df['C'].unique())
    ax.set_xticks(unique_c)
    
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    
    # Labels & Title
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Regularization Parameter C (log scale)")
    plt.title("Logistic Regression: Regularization C")
    plt.grid(True, linestyle='--', alpha=0.5)

    for split in ["Train Accuracy", "Test Accuracy"]:
        subset = df_melted[df_melted["Split"] == split]
        for _, row in subset.iterrows():
            ax.text(
                row['C'], 
                row['Accuracy'] + 0.005, 
                f"{row['Accuracy']:.3f}", 
                ha='center', va='bottom', fontsize=9, color='black'
            )

    plt.show()

def plot_svm(df):
    """
    Plots results for Support Vector Machine experiments.
    """
    df = df.copy()
    
    def parse_svm_row(row):
        c_match = re.search(r'C=([0-9.]+)', row)
        c_val = float(c_match.group(1)) if c_match else 0.0
        name = row.split('(')[0].strip()
        return pd.Series([name, c_val])

    df[['Method', 'C']] = df['Descriptor'].apply(parse_svm_row)

    df_melted = df.melt(
        id_vars=["Method", "C"], 
        value_vars=["Train Accuracy", "Test Accuracy"], 
        var_name="Split", 
        value_name="Accuracy"
    )
    df_melted = df_melted.sort_values(by=["Method", "C"])

    g = sns.FacetGrid(df_melted, col="Method", height=6, aspect=1, sharey=True)

    g.map_dataframe(
        sns.lineplot, 
        x="C", 
        y="Accuracy", 
        hue="Split", 
        palette="viridis", 
        marker="o",
        linewidth=2.5,
        markersize=8
    )

    unique_c = sorted(df['C'].unique())

    for ax in g.axes.flat:
        ax.set_xscale('log')
        ax.set_xticks(unique_c)
        ax.get_xaxis().set_major_formatter(ScalarFormatter()) # No scientific notation
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Regularization C (log scale)")

    methods = sorted(df_melted['Method'].unique())

    for ax, method in zip(g.axes.flat, methods):
        subset = df_melted[df_melted['Method'] == method]
        
        for _, row in subset.iterrows():
            # Logic: Test Bottom, Train Top
            if "Test" in row['Split']:
                offset = -0.02
                valign = 'top'
            else:
                offset = +0.02
                valign = 'bottom'

            ax.text(
                row['C'], 
                row['Accuracy'] + offset, 
                f"{row['Accuracy']:.3f}", 
                ha='center', 
                va=valign, 
                fontsize=9, 
                color='black'
            )

    g.add_legend(title="Dataset Split")
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle("SVM: Regularization C", fontsize=16)

    plt.show()

def plot_svm_rbf_gamma(df):
    """
    Plots results for Support Vector Machine experiments with RBF kernel.
    """
    df = df.copy()

    def parse_gamma_row(row):
        match = re.search(r'Gamma=([A-Za-z0-9.]+)', row)
        return match.group(1) if match else "unknown"

    df['Gamma'] = df['Descriptor'].apply(parse_gamma_row)

    df_melted = df.melt(
        id_vars=["Gamma"], 
        value_vars=["Train Accuracy", "Test Accuracy"], 
        var_name="Split", 
        value_name="Accuracy"
    )

    custom_order = ["scale", "auto", "0.1", "1", "10", "100"]
    present_order = [x for x in custom_order if x in df['Gamma'].unique()]

    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=df_melted,
        x="Gamma",
        y="Accuracy",
        hue="Split",
        order=present_order,
        palette="viridis"
    )

    # --- 1. Add Labels to the Bars ---
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

    # --- 2. Calculate Dynamic Header Room ---
    # Find max height to ensure red text fits
    ax.set_ylim(0, 1.2) # Set limit to 135% of max bar height

    train_bars = ax.containers[0]
    test_bars = ax.containers[1]

    for b1, b2 in zip(train_bars, test_bars):
        x1 = b1.get_x() + b1.get_width() / 2
        y1 = b1.get_height()
        x2 = b2.get_x() + b2.get_width() / 2
        y2 = b2.get_height()
        
        gap = y1 - y2
        
        # Draw Line
        ax.plot([x1, x2], [y1, y2], color='#c44e52', linestyle='--', marker='o', linewidth=1.5, markersize=4)
        
        # Draw Text
        mid_x = (x1 + x2) / 2
        
        # Position text dynamically based on max height (pushes it up above bar labels)
        mid_y = max(y1, y2) + 0.06
        
        ax.text(
            mid_x, 
            mid_y, 
            f"+{gap:.3f}", 
            color='#c44e52', 
            ha='center', 
            va='bottom', 
            fontsize=9, 
            fontweight='bold',
            # Add white background box
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
        )

    plt.title("SVM-RBF: Gamma Parameter")
    plt.ylabel("Accuracy")
    plt.xlabel("Gamma Parameter")
    
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