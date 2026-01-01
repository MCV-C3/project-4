import sys
import os
from argparse import Namespace

# Get the absolute path to the project root (C3/Alvaro/project-4)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add both the root AND Week1 to the system path
if root_path not in sys.path:
    sys.path.append(root_path)
    sys.path.append(os.path.join(root_path, 'Week1'))
    sys.path.append(os.path.join(root_path, 'Week2'))

# Now Python can find 'Week1' (as a package) AND 'bovw' (as a standalone module)
from Week1.final_evaluation import evaluate_model

from Week2.main_part1 import main as main_small_images
import Week2.train_eval_best_svm as best_svm
from Week2.main_part2 import main as main_patches 
from Week2.bow_part2 import main as main_bovw

# Week 1
evaluate_model(config_path="configs/baselines/week1_LR.json", output_dir="results/week1_LR")

# Week 2
# Small images
main_small_images(Namespace(config="configs/baselines/week2_small_images.json", dry_run=False))

# SVM
train_ds, test_ds, model = best_svm.load_resources(32, "configs/baselines/week2_small_images.json", "experiments/week2_small_images/week2_small_images.pth") 
mlp_metrics, (mlp_fig_cm, mlp_fig_roc) = best_svm.run_mlp_eval(test_ds, model, output_dir="results/")
svm_metrics, (svm_fig_cm, svm_fig_roc) = best_svm.run_svm_eval(train_ds, test_ds, model, C=7, kernel='linear', layer_index='output',  output_dir="results/")

# Patches 
main_patches(Namespace(config="configs/baselines/week2_dense_desc.json", dry_run=False))

# BoVW
main_bovw(Namespace(config="configs/baselines/week2_bovw.json", dry_run=False))

