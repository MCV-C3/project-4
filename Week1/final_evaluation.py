import os

# Limit threads to avoid threadpoolctl/OpenMP crashing on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import threadpoolctl
# Monkeypatch threadpoolctl to avoid crashing on Windows/Conda when getting library versions
class _DummyThreadpoolLimits:
    def __init__(self, limits=None, user_api=None):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Overwrite the class/function in the module
threadpoolctl.threadpool_limits = _DummyThreadpoolLimits

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize

import main
# Patch wandb to avoid AttributeError if it's not initialized or old version
try:
    _ = main.wandb.run
except AttributeError:
    main.wandb.run = None

from main import Dataset, process_dataset, extract_bovw_histograms, get_classifier, get_scaler, BOVW, train, check_descriptors_exist



import cv2

# Local implementation of train/test to support custom descriptors_root for debug
from main import fit_codebook_batched, fit_pca_batched, cross_validation

def train_model(dataset, bovw, config, load_descriptors=True, descriptors_root="./computed_descriptors"):
    
    # Get Descriptors
    print("Processing Train Data...")
    train_paths, train_labels = process_dataset(dataset, bovw, split_name="train", load_from_disk=load_descriptors, descriptors_root=descriptors_root)
    
    # Fit codebook by batches (RAM saving)
    if bovw.use_pca:
        fit_pca_batched(bovw, train_paths, batch_size=config["codebook_batch_size"])
        
    fit_codebook_batched(bovw, train_paths, batch_size=config["codebook_batch_size"])

    print("Computing BoVW histograms [Train]...")
    bovw_histograms = extract_bovw_histograms(bovw=bovw, descriptor_paths=train_paths, spatial_pyramid=config["spatial_pyramid"])
    
    print(f"Fitting the classifier: {config['classifier']}...")
    classifier = get_classifier(config["classifier"], config)
    
    # Scaling
    scaler = get_scaler(config.get("scaler"))
    if scaler:
        print(f"Fitting Scaler: {config['scaler']}...")
        bovw_histograms = scaler.fit_transform(bovw_histograms)
    
    # Cross Validation
    print("Performing Cross-Validation...")
    try:
        scores = cross_validation(classifier, bovw_histograms, train_labels, cv=5)
    except Exception as e:
        print(f"CV Failed (possibly too few samples in debug?): {e}")
        scores = 0

    classifier.fit(bovw_histograms, train_labels)

    train_acc = accuracy_score(y_true=train_labels, y_pred=classifier.predict(bovw_histograms))
    print(f"Accuracy on Phase [Train]: {train_acc:.4f}")
    
    return bovw, classifier, scaler, train_acc, scores

def evaluate_model(config_path="configs/final_model.json", output_dir="results/final_evaluation", debug=False):
    """
    Loads config, trains model, and performs comprehensive evaluation on Test Set.
    Args:
        debug (bool): If True, runs on a small subset of data for quick testing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        configs = json.load(f)
    
    # Assuming the first config in the file is the final model
    config = configs[0]["config"]
    name = configs[0]["name"]
    
    descriptors_root = "./computed_descriptors"
    if debug:
        print("!!! DEBUG MODE ENABLED: Using 50 samples per split !!!")
        config["max_samples_train"] = 50
        config["max_samples_test"] = 50
        descriptors_root = "./computed_descriptors_debug"
        # Optional: change output dir to avoid overwriting full run?
        # output_dir = os.path.join(output_dir, "debug")
        # os.makedirs(output_dir, exist_ok=True)

    print(f"Evaluating Final Model: {name}")
    print(f"Configuration: {config}")

    # 2. Setup Data & Pipeline
    DATA_PATH = config.get("data_path", "../data/MIT_split/")
    
    # Load Datasets
    print("Loading datasets...")
    data_train = Dataset(os.path.join(DATA_PATH, "train"), config.get("max_samples_train"))
    data_test = Dataset(os.path.join(DATA_PATH, "test"), config.get("max_samples_test"))
    
    # Initialize BOVW
    bovw = BOVW(
        detector_type=config["detector"],
        dense=config["dense"],
        step_size=config["step_size"],
        scale=config["scale"],
        codebook_size=config["codebook_size"],
        levels=config["levels"],
        detector_kwargs=config.get("detector_kwargs", {}),
        random_state=config.get("seed", 42),
        use_pca=config.get("use_pca", False),
        n_components=config.get("pca_components", 64),
        normalization=config.get("normalization", "l2")
    )

    # Check/Load Descriptors
    load_descriptors = False
    if check_descriptors_exist(bovw, base_dir=descriptors_root):
        print("Descriptors found on disk. Loading...")
        load_descriptors = True
    else:
        print("Descriptors not found. Computing...")

    # 3. Train Model (on Train Set) OR Load from Disk
    model_path = os.path.join(output_dir, "model.pkl")
    
    if os.path.exists(model_path) and not debug: # Don't load full model in debug mode unless we want to test that logic, but usually debug means we want to run quick.
        print(f"Found cached model at {model_path}. Loading...")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            bovw = saved_data["bovw"]
            classifier = saved_data["classifier"]
            scaler = saved_data["scaler"]
            
            # Re-initialize detector because it wasn't pickled
            if bovw.detector_type == 'SIFT':
                bovw.detector = cv2.SIFT_create(**bovw.detector_kwargs)
            elif bovw.detector_type == 'AKAZE':
                bovw.detector = cv2.AKAZE_create(**bovw.detector_kwargs)
            elif bovw.detector_type == 'ORB':
                bovw.detector = cv2.ORB_create(**bovw.detector_kwargs)
                
            print("Model loaded successfully.")
    else:   
        # Using local train_model function
        print("Model not found (or in Debug mode). Training model...")
        bovw, classifier, scaler, train_acc, _ = train_model(data_train, bovw, config, load_descriptors=load_descriptors, descriptors_root=descriptors_root)
        
        # Save Model
        save_path = model_path
        if debug:
            save_path = os.path.join(output_dir, "model_debug.pkl")
            
        print(f"Saving model to {save_path}...")
        
        # Temporarily remove detector (cv2 object) to allow pickling
        temp_detector = bovw.detector
        bovw.detector = None
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                "bovw": bovw,
                "classifier": classifier,
                "scaler": scaler
            }, f)
        
        # Restore detector
        bovw.detector = temp_detector
        print("Model saved.")
    
    # 4. Predict on Test Set
    print("Processing Test Set...")
    test_paths, test_labels = process_dataset(data_test, bovw, split_name="test", load_from_disk=load_descriptors, descriptors_root=descriptors_root)
    
    print("Computing BoVW histograms (Test)...")
    X_test = extract_bovw_histograms(bovw, test_paths, spatial_pyramid=config["spatial_pyramid"])
    
    if scaler:
        print("Applying Scaler transform...")
        X_test = scaler.transform(X_test)
        
    print("Predicting...")
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test) # Needed for ROC AUC

    # 5. Metrics
    print("\n" + "="*30)
    print("FINAL EVALUATION RESULTS")
    print("="*30)
    
    # Quantitative
    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred, average='macro')
    recall = recall_score(test_labels, y_pred, average='macro')
    f1 = f1_score(test_labels, y_pred, average='macro')
    
    # ROC AUC (Multi-class)
    # Binarize labels for ROC AUC
    classes = sorted(list(set(test_labels)))
    y_test_bin = label_binarize(test_labels, classes=classes)
    n_classes = y_test_bin.shape[1]
    
    roc_auc = None
    try:
        # Calculate ROC AUC
        if len(classes) == 2:
            # Binary case
             roc_auc = roc_auc_score(test_labels, y_prob[:, 1])
        else:
            # Multi-class
            roc_auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')
        
        # Plot ROC Curve
        from sklearn.metrics import roc_curve, auc
        
        plt.figure(figsize=(10, 8))
        
        # Plot for each class
        if len(classes) > 2:
            fpr = dict()
            tpr = dict()
            roc_auc_dict = dict()
            
            for i in range(n_classes):
                # Ensure we have enough data points for ROC
                if np.sum(y_test_bin[:, i]) > 0:
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                    roc_auc_dict[i] = auc(fpr[i], tpr[i])
                    plt.plot(fpr[i], tpr[i], lw=1, alpha=0.5, label=f'Class {classes[i]} (AUC = {roc_auc_dict[i]:.2f})')
                    
            # Compute macro-average ROC curve and ROC area
            # Aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes) if i in fpr]))

            # Interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            valid_classes = 0
            for i in range(n_classes):
                if i in fpr:
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                    valid_classes += 1
            
            if valid_classes > 0:
                mean_tpr /= valid_classes

            roc_auc_macro = auc(all_fpr, mean_tpr)

            plt.plot(all_fpr, mean_tpr,
                     label=f'Macro-average ROC curve (area = {roc_auc_macro:.2f})',
                     color='black', linestyle=':', linewidth=3)

        else:
            # Binary Case plotting
            fpr, tpr, _ = roc_curve(test_labels, y_prob[:, 1])
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, "roc_curve.png"))
        plt.close()
        print("ROC Curve saved.")

    except Exception as e:
        print(f"Warning: ROC AUC or Plot could not be calculated: {e}")


    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC:   {roc_auc:.4f}")
    
    # Save Metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # 6. Qualitative Visualizations
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    print("Confusion Matrix saved.")

    # Misclassified Examples
    print("Analyzing misclassified examples...")
    misclassified_indices = np.where(np.array(test_labels) != y_pred)[0]
    
    # Identify unique classes to map back if needed, or just use indices
    # We will save a few examples
    num_examples = 5
    if len(misclassified_indices) > 0:
        fig, axes = plt.subplots(1, min(len(misclassified_indices), num_examples), figsize=(15, 5))
        if len(misclassified_indices) == 1:
            axes = [axes]
        elif num_examples == 1:
             axes = [axes]
            
        # Ensure axes is iterable if we requested more than 1 but got 1 or more
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        for i, idx in enumerate(misclassified_indices[:num_examples]):
            # Get original image
            img, true_label = data_test[idx]
            pred_label = y_pred[idx]
            
            axes[i].imshow(img)
            axes[i].set_title(f"True: {true_label} | Pred: {pred_label}")
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "misclassified_examples.png"))
        plt.close()
        print(f"Top {num_examples} misclassified examples saved.")
    else:
        print("Perfect classification! No misclassified examples to show.")

    print(f"\nEvaluation Complete. Results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run with small subset")
    args = parser.parse_args()
    
    evaluate_model(debug=args.debug)

