
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
        
        current_config = copy.deepcopy(baseline)
        full_experiment_name = f"{sweep_name}/{config_name}"
        current_config['experiment_name'] = full_experiment_name
        
        # Update/Override parameters from variation
        for key, value in var.items():
            if key != 'name':
                current_config[key] = value

        temp_config_path = os.path.join(sweep_dir, f"temp_{config_name}.json")
        with open(temp_config_path, 'w') as f:
            json.dump(current_config, f, indent=4)
            
        # Ensure it calls the correct main script
        cmd = [sys.executable, "main.py", "--config", temp_config_path]
        
        if dry_run:
            cmd.append("--dry-run")
        
        try:
            subprocess.run(cmd, check=True)
            
            # The output directory logic in main.py usually matches the experiment name
            results_path = os.path.join("experiments", full_experiment_name, "results.json")
            
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    res = json.load(f)
                    summary_item = {
                        "experiment_name": res.get("experiment_name"),
                        "unfreeze_depth": res.get("unfreeze_depth"),
                        "lr": res.get("lr"),
                        "train_accuracy": res.get("train_accuracy"),
                        "best_test_accuracy": res.get("best_test_accuracy"),
                        "final_test_accuracy": res.get("final_test_accuracy"),
                        "final_test_loss": res.get("final_test_loss")
                    }
                    all_results.append(summary_item)
            else:
                print(f"Warning: No results.json found for {config_name} at {results_path}")

        except subprocess.CalledProcessError as e:
            print(f"Experiment {config_name} failed with error: {e}")
        finally:
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
        print(f"{'Experiment':<30} | {'Depth':<10} | {'LR':<10} | {'Best Acc':<15}")
        print("-" * 75)

        for res in all_results:
            display_name = res.get('experiment_name', '').split('/')[-1]
            unfreeze = res.get('unfreeze_depth', 0)
            lr_val = res.get('lr', 0.0)
            accuracy = float(res.get('best_test_accuracy') or 0.0) 
            print(f"{display_name:<30} | {unfreeze:<10} | {lr_val:<10.2e} | {accuracy:<15.4f}")
    else:
        print("\nSweep completed but no results collected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sweep_layers.json", help="Path to sweep config")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (1 batch per epoch)")
    args = parser.parse_args()
    
    run_sweep(args.config, args.dry_run)
