
import os
import json
import subprocess
import csv
import copy
import argparse
import sys

def run_sweep(sweep_config_path, dry_run=False):
    with open(sweep_config_path, 'r') as f:
        sweep_data = json.load(f)

    baseline = sweep_data['baseline']
    variations = sweep_data['variations']
    sweep_name = sweep_data.get('sweep_name', 'default_sweep')
    
    all_results = []
    
    # Ensure sweep directory exists
    sweep_dir = os.path.join("experiments", sweep_name)
    os.makedirs(sweep_dir, exist_ok=True)

    print(f"Starting sweep '{sweep_name}' with {len(variations)} variations...")
    print(f"Baseline config: {baseline}")

    for i, var in enumerate(variations):
        config_name = var['name']
        print(f"\n[{i+1}/{len(variations)}] Running experiment: {config_name}")
        
        # Merge baseline with variation
        current_config = copy.deepcopy(baseline)
        # Use nested path for experiment name so main.py creates it in the subfolder
        full_experiment_name = f"{sweep_name}/{config_name}"
        current_config['experiment_name'] = full_experiment_name
        
        # Update/Override parameters
        # Simple recursive merge for model_params
        if 'model_params' in var:
            current_config['model_params'].update(var['model_params'])
        
        # Update top-level keys
        for key, value in var.items():
            if key not in ['name', 'model_params']:
                current_config[key] = value

        # Write temp config
        # We can save temp config in the sweep dir to keep root clean
        temp_config_path = os.path.join(sweep_dir, f"temp_{config_name}.json")
        with open(temp_config_path, 'w') as f:
            json.dump(current_config, f, indent=4)
            
        # Run main.py
        # Using sys.executable to ensure we use the same python interpreter
        cmd = [sys.executable, "main_part1.py", "--config", temp_config_path]
        
        # Optional: Add dry-run for testing logic
        if dry_run:
            cmd.append("--dry-run")
        
        try:
            subprocess.run(cmd, check=True)
            
            # Read results
            # main.py will create experiments/sweep_name/config_name/results.json
            results_path = os.path.join("experiments", full_experiment_name, "results.json")
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    try:
                        res = json.load(f)
                        # clean up model_params for csv
                        res['hidden_layers'] = str(res['model_params'].get('hidden_layers'))
                        res['dropout'] = res['model_params'].get('dropout')
                        res['activation'] = res['model_params'].get('activation')
                        del res['model_params']
                        all_results.append(res)
                    except json.JSONDecodeError:
                         print(f"Error decoding results.json for {config_name}")
            else:
                print(f"Warning: No results.json found for {config_name} at {results_path}")

        except subprocess.CalledProcessError as e:
            print(f"Experiment {config_name} failed with error: {e}")
        finally:
            # Clean up temp config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

    # Save aggregated results in the sweep directory
    if all_results:
        csv_path = os.path.join(sweep_dir, "sweep_results.csv")
        keys = all_results[0].keys()
        
        with open(csv_path, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_results)
            
        print(f"\nSweep completed. Results saved to {csv_path}")
        print("\nSummary:")
        print(f"{'Experiment':<30} | {'Layers':<30} | {'Best Val Acc':<15} | {'Final Val Acc':<15}")
        print("-" * 100)
        for res in all_results:
            # Strip the folder prefix for cleaner display if desired
            display_name = res.get('experiment_name', '').split('/')[-1]
            print(f"{display_name:<30} | {res.get('hidden_layers', ''):<30} | {res.get('best_val_accuracy', 0):<15.4f} | {res.get('final_val_accuracy', 0):<15.4f}")
    else:
        print("\nSweep completed but no results collected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sweep_layers.json", help="Path to sweep config")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (1 batch per epoch)")
    args = parser.parse_args()
    
    run_sweep(args.config, args.dry_run)
