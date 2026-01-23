import os
import json
import argparse
import glob
import csv

import matplotlib.pyplot as plt
import numpy as np


# set size parameters of the plots to be big enought to be slide-readable
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})



def load_results(sweep_dir):
    results = []
    # Search for all experiment_results.json files in the sweep directory
    pattern = os.path.join(sweep_dir, "*", "experiment_results.json")
    files = glob.glob(pattern)

    if not files:
        print(f"No results found in {sweep_dir}")
        return []

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            experiment_name = data.get('configuration', {}).get(
                'experiment_name', 'Unknown')
            metrics = data.get('metrics', {})

            # Extract relevant metrics
            params = metrics.get('model_params')
            accuracy = metrics.get('final_val_accuracy')
            if accuracy is None:
                accuracy = metrics.get('final_test_accuracy')

            if params is not None and accuracy is not None:
                results.append({
                    'name': experiment_name,
                    'params': params,
                    'accuracy': accuracy
                })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return results


def plot_results(results, output_file):
    if not results:
        print("No data to plot.")
        return

    params = np.array([r["params"] for r in results], dtype=float)
    accuracies = np.array([r["accuracy"] for r in results], dtype=float)
    names = np.array([r["name"] for r in results], dtype=object)

    # --- Compute extra fields (kept from your code) ---
    max_params = float(params.max()) if len(params) else 1.0
    for r in results:
        norm_param = r["params"] / max_params
        r["norm_param"] = norm_param
        r["distance"] = float(np.sqrt((1 - r["accuracy"]) ** 2 + norm_param ** 2))

    # --- Pareto frontier (same logic as yours) ---
    sorted_indices = np.argsort(params)
    sorted_params = params[sorted_indices]
    sorted_accs = accuracies[sorted_indices]

    current_max_acc = -np.inf
    pareto_params, pareto_accs = [], []

    for p, a in zip(sorted_params, sorted_accs):
        if a > current_max_acc:
            current_max_acc = a
            pareto_params.append(p)
            pareto_accs.append(a)

    pareto_params = np.array(pareto_params, dtype=float)
    pareto_accs = np.array(pareto_accs, dtype=float)

    # Boolean mask of which original points are on the frontier
    # (safe even if duplicate params/accs exist)
    pareto_set = set(zip(pareto_params.tolist(), pareto_accs.tolist()))
    pareto_mask = np.array([(p, a) in pareto_set for p, a in zip(params, accuracies)], dtype=bool)

    # --- Helper to draw the base plot, optionally with frontier-only annotations ---
    def _draw(annotate_frontier_only: bool, out_path: str):
        plt.figure(figsize=(10, 8))

        plt.scatter(params, accuracies, alpha=0.7)
        plt.plot(pareto_params, pareto_accs, "r--", label="Pareto Frontier", alpha=0.5)

        if annotate_frontier_only:
            frontier_idx = np.where(pareto_mask)[0]
            for i in frontier_idx:
                if names[i] not in ["MobileNetV2Exp_Halved_Thin3", "MobileNetV1_Halved_Thin2"]:
                    plt.annotate(
                        str(names[i]),
                        (params[i], accuracies[i]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=10,
                    )


        plt.xscale("log")
        plt.xlim(200, 1e7)      # X axis: [200, 10^7]
        plt.ylim(0.65, 0.9)   # Y axis: [0.5, 1.0]

        plt.xlabel("Number of Weights (Parameters)")
        plt.ylabel("Test Accuracy")
        plt.title("Model Accuracy vs. Number of Weights")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Plot saved to {out_path}")

    # --- Save 2 plots ---
    base, ext = os.path.splitext(output_file)
    if not ext:
        ext = ".png"

    plain_path = f"{base}_plain{ext}"
    annotated_path = f"{base}_frontier_annotated{ext}"

    _draw(annotate_frontier_only=False, out_path=plain_path)
    _draw(annotate_frontier_only=True, out_path=annotated_path)

    # Sort results by distance (closest to ideal first)
    sorted_by_dist = sorted(results, key=lambda x: x['distance'])

    # Save to CSV
    # Derive CSV filename from output filename to include the suffix
    csv_filename = os.path.basename(output_file).replace(
        'accuracy_vs_weights', 'model_comparison').replace('.png', '.csv')
    csv_file = os.path.join(os.path.dirname(output_file), csv_filename)
    try:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Model Name', 'Parameters', 'Accuracy',
                            'Distance to Ideal (Acc=1, Params=0)'])
            for r in sorted_by_dist:
                writer.writerow(
                    [r['name'], r['params'], f"{r['accuracy']:.4f}", f"{r['distance']:.4f}"])
        print(f"Comparison table saved to {csv_file}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    # Print top 5 models by distance
    print("\nTop 5 Models by Distance to Ideal (Acc=1, Params=0):")
    print(f"{'Model Name':<25} | {'Params':<10} | {'Accuracy':<10} | {'Distance':<10}")
    print("-" * 65)
    for r in sorted_by_dist[:5]:
        print(
            f"{r['name']:<25} | {r['params']:<10} | {r['accuracy']:.4f}     | {r['distance']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot Accuracy vs Weights for a Sweep')
    parser.add_argument('sweep_dirs', type=str, nargs='+',
                        help='Paths to the sweep directories (e.g., results/sweeps/hbuck2cj)')

    args = parser.parse_args()

    all_results = []
    sweep_ids = []

    for sweep_dir in args.sweep_dirs:
        print(f"Loading results from {sweep_dir}...")
        results = load_results(sweep_dir)
        print(f"Found {len(results)} experiments in {sweep_dir}.")
        all_results.extend(results)

        # Extract sweep ID from path for filename
        sweep_id = os.path.basename(os.path.normpath(sweep_dir))
        sweep_ids.append(sweep_id)

    if not all_results:
        print("No results found in any directory.")
        return

    # Create suffix from sweep IDs
    suffix = "_" + "_".join(sweep_ids)

    # Save output in the first directory provided
    first_dir = args.sweep_dirs[0]
    output_file = os.path.join(first_dir, f'accuracy_vs_weights{suffix}.png')

    plot_results(all_results, output_file)


if __name__ == "__main__":
    main()
