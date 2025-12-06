import sys
from unittest.mock import MagicMock

# MOCK THREADPOOLCTL to avoid AttributeError and ctypes crashes
# The environment seems to have a broken threadpoolctl/OpenBLAS interaction
mock_tpc = MagicMock()
mock_tpc.__version__ = "3.0.0"
class MockLimits:
    def __init__(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def register(self, *args): pass
    def unregister(self, *args): pass
    def get_original_limits(self): return {}
mock_tpc.threadpool_limits = MockLimits
mock_tpc.threadpool_info = lambda: []
sys.modules['threadpoolctl'] = mock_tpc

import pandas as pd
import os
import itertools
from datetime import datetime
from main import run_experiment

def run_sweep():
    # --- Sweep Configuration ---
    
    # Grid Search Parameters
    
    # 1. Classifiers
    classifiers = ["LogisticRegression", "SVM-Linear", "SVM-RBF", "SVM-HistogramIntersection"]
    # classifiers = ["SVM-Linear"] # For testing
    
    # 2. C Parameter
    # c_params = [0.1, 1.0, 10.0, 100.0]
    c_params = [1.0, 10.0] # Reduced as requested
    
    # 3. Gamma (Only for RBF)
    gamma_params = ['scale', 'auto']

    # 4. Detectors
    detectors_base = ["SIFT", "ORB", "AKAZE"]

    # 5. Spatial Pyramid
    pyramid_options = [False, True]
    
    # 6. PCA Components
    pca_options = [None, 64] # None = No PCA

    results = []
    
    # --- Generate Combinations ---
    
    # Iterate Detectors
    for detector in detectors_base:
        
        # Determine Dense options for this detector
        if detector == "SIFT":
            dense_options = [False, True] 
        else:
            dense_options = [False]
            
        for dense in dense_options:
            
            for pyramid in pyramid_options:
                
                for pca_val in pca_options:

                    for classifier in classifiers:
                        # Hardcoded skip for existing results
                        # We only skip if pca is None, because previous runs didn't have PCA
                        if detector == "SIFT" and dense is False and pyramid is False and pca_val is None and classifier in ["LogisticRegression", "SVM-Linear"]:
                            print(f"Skipping {detector} Dense={dense} Pyramid={pyramid} PCA={pca_val} {classifier} (Hardcoded skip)")
                            continue
                        
                        # Determine Gamma options
                        current_gammas = gamma_params if classifier == "SVM-RBF" else ['scale'] 
                        
                        for c_val in c_params:
                            for gamma_val in current_gammas:
                                
                                print(f"\n=== Running Sweep: Det={detector}, Dense={dense}, Pyramid={pyramid}, PCA={pca_val}, Clf={classifier}, C={c_val}, Gamma={gamma_val} ===")
                                
                                try:
                                    # Run Experiment
                                    metrics = run_experiment(
                                        detector=detector,
                                        dense=dense,
                                        classifier=classifier,
                                        c_param=c_val,
                                        gamma_param=gamma_val,
                                        codebook_size=1024, # Fixed for now
                                        spatial_pyramid=pyramid,
                                        pca_components=pca_val,
                                        use_wandb=True # Enable WandB for full run
                                    )
                                    # Note: pass use_wandb=True if you want to log every run
                                    
                                    # Append extra info
                                    metrics["spatial_pyramid"] = pyramid
                                    metrics["pca_components"] = pca_val
                                    metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    results.append(metrics)
                                    
                                except Exception as e:
                                    import traceback
                                    traceback.print_exc()
                                    print(f"!!! Error in run: {e}")
                                    # Log failure
                                    results.append({
                                        "detector": detector,
                                        "dense": dense,
                                        "classifier": classifier,
                                        "C": c_val,
                                        "gamma": gamma_val,
                                        "spatial_pyramid": pyramid,
                                        "pca_components": pca_val,
                                        "test_accuracy": 0.0,
                                        "train_accuracy": 0.0,
                                        "cv_accuracy": 0.0,
                                        "time": 0.0,
                                        "error": str(e)
                                    })
        
                                # Save intermediate results in case of crash
                                df = pd.DataFrame(results)
                                df.to_csv("sweep_results.csv", index=False)

    # --- Save Final Results ---
    df_final = pd.DataFrame(results)
    output_file = "sweep_results.csv"
    df_final.to_csv(output_file, index=False)
    print(f"\nSweep Completed. Results saved to {output_file}")


if __name__ == "__main__":
    run_sweep()
