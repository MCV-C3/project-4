import os
import json
import matplotlib.pyplot as plt
import argparse
import glob
import numpy as np
import csv


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

    params = [r['params'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    names = [r['name'] for r in results]

    plt.figure(figsize=(10, 8))
    # X = Weights, Y = Accuracy
    plt.scatter(params, accuracies, color='blue', alpha=0.7)

    # Annotate points
    for i, name in enumerate(names):
        plt.annotate(name, (params[i], accuracies[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.xscale('log')  # Log scale for parameters makes sense on X
    plt.xlabel('Number of Weights (Parameters)')
    plt.ylabel('Test Accuracy')
    plt.title('Model Accuracy vs. Number of Weights')
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Calculate Distances to Ideal (Acc=1, Params=0)
    # Normalize params to [0, 1] relative to the max observed to make the distance meaningful
    max_params = max(params) if params else 1

    # Distance = sqrt((1 - acc)^2 + (params/max_params)^2)
    distances = []
    for r in results:
        norm_param = r['params'] / max_params
        dist = np.sqrt((1 - r['accuracy'])**2 + norm_param**2)
        r['distance'] = dist
        r['norm_param'] = norm_param
        distances.append(dist)

    # Highlight the Pareto frontier (simple version: max accuracy for params <= p)
    # Sort by params
    sorted_indices = np.argsort(params)
    sorted_params = np.array(params)[sorted_indices]
    sorted_accs = np.array(accuracies)[sorted_indices]

    current_max_acc = -1
    pareto_params = []
    pareto_accs = []

    for p, a in zip(sorted_params, sorted_accs):
        if a > current_max_acc:
            current_max_acc = a
            pareto_params.append(p)
            pareto_accs.append(a)

    plt.plot(pareto_params, pareto_accs, 'r--',
             label='Pareto Frontier', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

    # Sort results by distance (closest to ideal first)
    sorted_by_dist = sorted(results, key=lambda x: x['distance'])

    # Save to CSV
    csv_file = os.path.join(os.path.dirname(
        output_file), 'model_comparison.csv')
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
    parser.add_argument('sweep_dir', type=str,
                        help='Path to the sweep directory (e.g., results/sweeps/hbuck2cj)')

    args = parser.parse_args()

    print(f"Loading results from {args.sweep_dir}...")
    results = load_results(args.sweep_dir)
    print(f"Found {len(results)} experiments.")

    output_file = os.path.join(args.sweep_dir, 'accuracy_vs_weights.png')
    plot_results(results, output_file)


if __name__ == "__main__":
    main()
