import os
import json
import argparse
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_dropout(folder_name):
    # Pattern: Drop{rate}_{method}_Unf{depth}
    # Example: Drop0.1_classifier_Unf0
    match = re.match(r"Drop([\d\.]+)_([a-z_]+)_Unf(\d+)", folder_name)
    if match:
        return {
            "dropout": float(match.group(1)),
            "method": match.group(2),
            "unfreeze_depth": int(match.group(3))
        }
    return None

def parse_norm(folder_name):
    # Pattern: Norm_{type}_Unf{depth}
    # Example: Norm_batch_Unf0
    match = re.match(r"Norm_([a-z]+)_Unf(\d+)", folder_name)
    if match:
        return {
            "norm": match.group(1),
            "unfreeze_depth": int(match.group(2))
        }
    return None

def parse_reg(folder_name):
    # Pattern: L1_{l1}_L2_{l2}
    # Example: L1_0.001_L2_0
    match = re.match(r"L1_([\d\.]+)_L2_([\d\.]+)", folder_name)
    if match:
        return {
            "l1_reg": float(match.group(1)),
            "l2_reg": float(match.group(2))
        }
    return None

def get_accuracy(folder_path):
    results_path = os.path.join(folder_path, "results.json")
    if not os.path.exists(results_path):
        # Try finding a file that looks like a result if exact name differs, 
        # but user specified results.json structure implicitly or explicitly.
        # Check standard names.
        return None
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
            # Check for best_test_accuracy, or fallback
            if "best_test_accuracy" in data:
                return data["best_test_accuracy"]
            # Fallback keys just in case
            elif "test_accuracy" in data:
                return data["test_accuracy"]
            elif "accuracy" in data:
                return data["accuracy"]
            else:
                return None
    except Exception as e:
        print(f"Error reading {results_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Plot heatmaps from experiment results.")
    parser.add_argument("folder", help="Path to the experiments folder (e.g., experiments/norm)")
    parser.add_argument("--output_dir", help="Directory to save plots. Defaults to the input folder.", default=None)
    args = parser.parse_args()

    root_folder = args.folder
    output_dir = args.output_dir if args.output_dir else root_folder
    
    if not os.path.isdir(root_folder):
        print(f"Error: {root_folder} is not a directory.")
        return

    subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
    subfolders.sort()

    data = []
    mode = None
    
    # Detect type based on first few valid subfolders
    for folder in subfolders:
        if parse_dropout(folder):
            mode = 'dropout'
            break
        elif parse_norm(folder):
            mode = 'norm'
            break
        elif parse_reg(folder):
            mode = 'reg'
            break
    
    if not mode:
        print("Could not detect experiment type from folder names (dropout, norm, or reg).")
        return

    print(f"Detected experiment mode: {mode}")

    for folder in subfolders:
        params = None
        if mode == 'dropout':
            params = parse_dropout(folder)
        elif mode == 'norm':
            params = parse_norm(folder)
        elif mode == 'reg':
            params = parse_reg(folder)
        
        if params:
            full_path = os.path.join(root_folder, folder)
            acc = get_accuracy(full_path)
            if acc is not None:
                params['best_test_accuracy'] = acc
                data.append(params)
            else:
                # Optionally warn
                # print(f"Warning: No accuracy found in {folder}")
                pass

    df = pd.DataFrame(data)
    
    if df.empty:
        print("No data found (or no results.json files with accuracy).")
        return

    sns.set_theme(font_scale=2.0)

    if mode == 'dropout':
        # Two heatmaps: classifier and feature_extractor
        methods = df['method'].unique()
        for method in methods:
            subset = df[df['method'] == method]
            if subset.empty:
                continue
            
            # Pivot: rows=unfreeze_depth, cols=dropout
            try:
                pivot = subset.pivot(index="unfreeze_depth", columns="dropout", values="best_test_accuracy")
                # Sort index and columns to be sure
                pivot = pivot.sort_index(axis=0).sort_index(axis=1)
                
                plt.figure(figsize=(16, 14))
                sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis", cbar_kws={'label': 'Test Accuracy'}, annot_kws={"size": 24})
                plt.title(f"Dropout Test Accuracy\nMethod: {method}", fontsize=32)
                plt.xlabel("Dropout Rate", fontsize=28)
                plt.ylabel("Unfreeze Depth", fontsize=28)
                plt.xticks(fontsize=26)
                plt.yticks(fontsize=26, rotation=0)
                
                out_path = os.path.join(output_dir, f"heatmap_dropout_{method}.png")
                plt.savefig(out_path)
                print(f"Saved plot to {out_path}")
                plt.close()
            except Exception as e:
                print(f"Error plotting dropout/{method}: {e}")
            
    elif mode == 'norm':
        try:
            pivot = df.pivot(index="unfreeze_depth", columns="norm", values="best_test_accuracy")
            pivot = pivot.sort_index(axis=0).sort_index(axis=1)

            plt.figure(figsize=(16, 14))
            sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis", cbar_kws={'label': 'Test Accuracy'}, annot_kws={"size": 24})
            plt.title("Normalization Test Accuracy", fontsize=32)
            plt.xlabel("Normalization Method", fontsize=28)
            plt.ylabel("Unfreeze Depth", fontsize=28)
            plt.xticks(fontsize=26)
            plt.yticks(fontsize=26, rotation=0)
            
            out_path = os.path.join(output_dir, "heatmap_norm.png")
            plt.savefig(out_path)
            print(f"Saved plot to {out_path}")
            plt.close()
        except Exception as e:
            print(f"Error plotting norm: {e}")
        
    elif mode == 'reg':
        try:
            pivot = df.pivot(index="l1_reg", columns="l2_reg", values="best_test_accuracy")
            pivot = pivot.sort_index(axis=0).sort_index(axis=1)

            plt.figure(figsize=(16, 14))
            sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis", cbar_kws={'label': 'Test Accuracy'}, annot_kws={"size": 24})
            plt.title("Regularization Test Accuracy", fontsize=32)
            plt.xlabel("L2 Regularization", fontsize=28)
            plt.ylabel("L1 Regularization", fontsize=28)
            plt.xticks(fontsize=26)
            plt.yticks(fontsize=26, rotation=0)
            
            out_path = os.path.join(output_dir, "heatmap_reg.png")
            plt.savefig(out_path)
            print(f"Saved plot to {out_path}")
            plt.close()
        except Exception as e:
            print(f"Error plotting reg: {e}")

if __name__ == "__main__":
    main()
