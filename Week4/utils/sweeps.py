import wandb

# TODO: POSTERIOR VERSION WILL USE THIS FUNCTION 
def configure_sweep(cfg):

    print("Sweep detected! Overriding config with sweep parameters.")
    for key, value in wandb.config.items():
        # Support nested keys like 'training.learning_rate' if your sweep defines them flat
        if '.' in key:
            section, subkey = key.split('.')
            if section in cfg:
                cfg[section][subkey] = value
        else:
            # If your sweep yaml structure matches your config structure exactly
            # Note: Standard sweeps usually flatten params. You might need custom logic 
            # if you have deep nesting, but this covers basic cases.
            pass 
            
    # Simpler approach if you flatten your sweep config:
    # Just update specific keys you know you are sweeping over:
    if 'learning_rate' in wandb.config:
        cfg['training']['learning_rate'] = wandb.config.learning_rate
    if 'batch_size' in wandb.config:
        cfg['data']['batch_size'] = wandb.config.batch_size
    # ... repeat for other sweepable params

    print(cfg)
    return cfg