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


def load_results(sweep_dir, pruning_mode=False, min_accuracy=0.6):
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

            metrics = data.get('metrics', {})

            # Extract relevant metrics
            accuracy = metrics.get('final_val_accuracy')
            if accuracy is None:
                accuracy = metrics.get('final_test_accuracy')
            efficiency = metrics.get('efficiency_score')

            if pruning_mode:
                # Use parent directory name as experiment name
                experiment_name = os.path.basename(os.path.dirname(file_path))
                params = metrics.get('final_nonzero_params')
                # Fallback if not found (e.g. baseline)
                if params is None:
                    params = metrics.get('model_params')
            else:
                experiment_name = data.get('configuration', {}).get(
                    'experiment_name', 'Unknown')
                params = metrics.get('model_params')

            if params is not None and accuracy is not None:
                if accuracy < min_accuracy:
                    continue
                res = {
                    'name': experiment_name,
                    'params': params,
                    'accuracy': accuracy,
                    'efficiency': efficiency
                }
                if pruning_mode:
                    res['non0_params'] = params
                    # Also store total params for reference if needed
                    res['total_params'] = metrics.get('model_params', params)
                results.append(res)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return results


def plot_results(results, output_file, pruning_mode=False, baseline_result=None):
    if not results:
        print("No data to plot.")
        return

    # If baseline provided, add it to results so it participates in Pareto, Distance, and Table
    if baseline_result:
        # Avoid duplicate addition if called multiple times (though here it's clean)
        results.append(baseline_result)

    # For pruning, 'params' in results is already set to non0_params by load_results
    params = np.array([r["params"] for r in results], dtype=float)
    accuracies = np.array([r["accuracy"] for r in results], dtype=float)
    names = np.array([r["name"] for r in results], dtype=object)

    # --- Compute extra fields ---
    # max_params = float(params.max()) if len(params) else 1.0
    # max_params = 100000
    max_params = 100000

    for r in results:
        norm_param = r["params"] / max_params
        r["norm_param"] = norm_param
        r["distance"] = float(
            np.sqrt((1 - r["accuracy"]) ** 2 + norm_param ** 2))

    # --- Pareto frontier (same logic) ---
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
    pareto_set = set(zip(pareto_params.tolist(), pareto_accs.tolist()))
    pareto_mask = np.array(
        [(p, a) in pareto_set for p, a in zip(params, accuracies)], dtype=bool)

    # --- Helper to draw the base plot ---
    def _draw(annotate_frontier_only: bool, out_path: str):
        plt.figure(figsize=(10, 8))

        plt.scatter(params, accuracies, alpha=0.7)
        plt.plot(pareto_params, pareto_accs, "r--",
                 label="Pareto Frontier", alpha=0.5)

        # Plot Baseline if provided (as distinct star)
        # Note: It is also plotted as a dot above because it's in `params`/`accuracies`.
        # That is acceptable (dot under star).
        if baseline_result:
            plt.scatter(baseline_result['params'], baseline_result['accuracy'],
                        color='black', marker='*', s=300, label=baseline_result['name'], zorder=10)
            if annotate_frontier_only:
                plt.annotate(
                    "Baseline",
                    (baseline_result['params'], baseline_result['accuracy']),
                    xytext=(5, -15),
                    textcoords="offset points",
                    fontsize=12,
                    color='black',
                    fontweight='bold'
                )

        if annotate_frontier_only:
            frontier_idx = np.where(pareto_mask)[0]
            for i in frontier_idx:
                # Filter out some specific names if needed, or keeping original filter
                # Also exclude "Baseline" since it has its own special annotation above
                if names[i] == "Baseline":
                    continue

                if names[i] not in ["MobileNetV2Exp_Halved_Thin3", "MobileNetV1_Halved_Thin2", "ShuffleNet_Pico"]:
                    plt.annotate(
                        str(names[i]),
                        (params[i], accuracies[i]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=10,
                    )

        plt.xscale("log")

        # Disable scientific notation for X axis
        from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
        ax = plt.gca()

        if pruning_mode:
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_minor_formatter(formatter)
            plt.xlabel("Number of Non-Zero Parameters")
        else:
            ax.xaxis.set_major_formatter(ScalarFormatter())
            plt.xlabel("Number of Weights (Parameters)")
            plt.xlim(200, 1e7)      # X axis: [200, 10^7]
            plt.ylim(0.65, 0.9)   # Y axis: [0.65, 0.9]

        plt.ylabel("Test Accuracy")
        plt.title("Model Accuracy vs. Number of Weights" +
                  (" (Pruning)" if pruning_mode else ""))
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

    # Sort results by accuracy (highest first)
    sorted_by_acc = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    # Save to CSV
    csv_filename = os.path.basename(output_file).replace(
        'accuracy_vs_weights', 'model_comparison').replace('.png', '.csv')
    csv_file = os.path.join(os.path.dirname(output_file), csv_filename)
    try:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            if pruning_mode:
                writer.writerow(['Model Name', 'Params (Total)', 'non0 Params', 'Accuracy',
                                'Distance to Ideal'])
                for r in sorted_by_acc:
                    writer.writerow(
                        [r['name'], r.get('total_params', 'N/A'), r['params'], f"{r['accuracy']:.4f}", f"{r['distance']:.4f}"])
            else:
                writer.writerow(['Model Name', 'Parameters', 'Accuracy',
                                'Distance to Ideal (Acc=1, Params=0)'])
                for r in sorted_by_acc:
                    writer.writerow(
                        [r['name'], r['params'], f"{r['accuracy']:.4f}", f"{r['distance']:.4f}"])

        print(f"Comparison table saved to {csv_file}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    # Print top 5 models by Accuracy
    print("\nTop 5 Models by Accuracy:")
    if pruning_mode:
        print(
            f"{'Model Name':<30} | {'Non0 Params':<12} | {'Accuracy':<10} | {'Distance':<10}")
        print("-" * 75)
        for r in sorted_by_acc[:5]:
            print(
                f"{r['name']:<30} | {r['params']:<12} | {r['accuracy']:.4f}     | {r['distance']:.4f}")
    else:
        print(
            f"{'Model Name':<25} | {'Params':<10} | {'Accuracy':<10} | {'Distance':<10}")
        print("-" * 65)
        for r in sorted_by_acc[:5]:
            print(
                f"{r['name']:<25} | {r['params']:<10} | {r['accuracy']:.4f}     | {r['distance']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot Accuracy vs Weights for a Sweep')
    parser.add_argument('sweep_dirs', type=str, nargs='+',
                        help='Paths to the sweep directories (e.g., results/sweeps/hbuck2cj)')
    parser.add_argument('--baseline_dir', type=str, required=False, default=None,
                        help='Path to a baseline model directory to plot for comparison')
    parser.add_argument('--pruning', action='store_true',
                        help='Enable pruning mode: use non-zero params for plots and table')

    args = parser.parse_args()

    # Determine min_accuracy filtering
    min_accuracy = 0.6
    if args.baseline_dir and "mobilenet" in args.baseline_dir.lower():
        min_accuracy = 0.8
        print("MobileNet baseline detected! Filtering results with accuracy < 0.8")
    else:
        print(
            f"Generic baseline (or none). Filtering results with accuracy < {min_accuracy}")

    all_results = []
    sweep_ids = []

    for sweep_dir in args.sweep_dirs:
        print(f"Loading results from {sweep_dir}...")
        results = load_results(
            sweep_dir, pruning_mode=args.pruning, min_accuracy=min_accuracy)
        print(f"Found {len(results)} experiments in {sweep_dir}.")
        all_results.extend(results)

        # Extract sweep ID from path for filename
        sweep_id = os.path.basename(os.path.normpath(sweep_dir))
        sweep_ids.append(sweep_id)

    if not all_results:
        print("No results found in any directory.")
        return

    baseline_result = None
    if args.baseline_dir:
        print(f"Loading baseline from {args.baseline_dir}...")
        # Baseline is a single directory, not a sweep, but load_results expects (parent/experiment_results.json)
        # Check if the baseline_dir DIRECTLY contains experiment_results.json or subfolders
        if os.path.exists(os.path.join(args.baseline_dir, "experiment_results.json")):
            # Create a list manually since load_results looks for */experiment_results.json
            # or we can write a simple loader for single file
            # Let's try to reuse load_results by passing the parent of baseline_dir if it matches pattern
            # Or just manual load:
            try:
                with open(os.path.join(args.baseline_dir, "experiment_results.json"), 'r') as f:
                    data = json.load(f)
                metrics = data.get('metrics', {})

                acc = metrics.get('final_val_accuracy') or metrics.get(
                    'final_test_accuracy')

                # Logic for params:
                # If pruning mode, we want the "starting" params.
                # Usually baseline is unpruned, so non0 = total.
                params = metrics.get('model_params')

                if acc is not None and params is not None:
                    baseline_result = {
                        'name': "Baseline",
                        'params': params,
                        'accuracy': acc,
                        'norm_param': 1.0  # Will be recomputed/ignored in plot
                    }
                    print(f"Loaded baseline: Acc={acc}, Params={params}")
            except Exception as e:
                print(f"Error loading baseline: {e}")
        else:
            # Try load_results if it has subfolders
            base_res = load_results(
                args.baseline_dir, pruning_mode=args.pruning)
            if base_res:
                baseline_result = base_res[0]  # Take the first one?
                print(
                    f"Loaded baseline from subfolder: {baseline_result['name']}")

    # Create suffix from sweep IDs
    suffix = "_" + "_".join(sweep_ids)
    if args.pruning:
        suffix += "_pruning"

    # Save output in the first directory provided
    first_dir = args.sweep_dirs[0]
    output_file = os.path.join(first_dir, f'accuracy_vs_weights{suffix}.png')

    plot_results(all_results, output_file, pruning_mode=args.pruning,
                 baseline_result=baseline_result)


if __name__ == "__main__":
    main()
