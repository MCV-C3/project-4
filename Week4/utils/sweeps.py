import wandb

def configure_sweep(cfg):
    """
    Updates the local config 'cfg' with parameters chosen by the WandB Sweep agent.
    Supports nested keys defined with dot notation (e.g., 'training.learning_rate').
    """
    print("\n>>> Sweep detected! Overriding config with sweep parameters:")
    
    # Iterate through all parameters injected by the sweep agent
    for key, value in wandb.config.items():
        # Handle nested keys (e.g., "training.learning_rate")
        if '.' in key:
            section, subkey = key.split('.', 1) # Split only on the first dot
            
            if section in cfg:
                # Update the specific value in the nested dictionary
                original_value = cfg[section].get(subkey, "New Key")
                cfg[section][subkey] = value
                print(f"    Overriding {section}.{subkey}: {original_value} -> {value}")
            else:
                print(f"    Warning: Section '{section}' not found in base config. Skipping {key}.")
        
        # Handle flat keys (if you ever add them)
        else:
            if key in cfg:
                print(f"    Overriding {key}: {cfg[key]} -> {value}")
                cfg[key] = value
            else:
                # If the key doesn't map to a config section, it might be meta-data, so we skip or log it
                pass

    print("-" * 50)
    return cfg
    