import os
import json

def get_best_in_folder(root_folder):
    """
    Finds the experiment configuration with the highest test accuracy within a root folder.
    
    Args:
        root_folder (str): Path to the folder containing experiment subdirectories.
        
    Returns:
        tuple: (best_config_name, best_accuracy, best_folder_path)
    """
    max_acc = -1.0
    best_config = None
    best_folder = None

    if not os.path.isdir(root_folder):
        return None, None, None

    for subdir in os.listdir(root_folder):
        dir_path = os.path.join(root_folder, subdir)
        if not os.path.isdir(dir_path):
            continue
        
        results_path = os.path.join(dir_path, "results.json")
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    data = json.load(f)
                    acc = data.get("best_test_accuracy", 0)
                    if acc > max_acc:
                        max_acc = acc
                        best_config = subdir
                        best_folder = dir_path
            except:
                pass
    return best_config, max_acc, best_folder

if __name__ == "__main__":
    print("--- Analysis ---")
    for exp_type in ['dropout', 'norm', 'reg', 'hyp_opt']:
        path = os.path.join("experiments", exp_type)
        config, acc, folder = get_best_in_folder(path)
        if config:
            print(f"Best {exp_type}: {config} (Acc: {acc})")
        else:
            print(f"Best {exp_type}: None found")
