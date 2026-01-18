import os
import gc
import glob
import time
import pickle
from typing import *

# Suppress joblib warning about physical cores
os.environ['LOKY_MAX_CPU_COUNT'] = "4"

import numpy as np
import tqdm
from PIL import Image
import wandb

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from bovw import BOVW


# ==========================================
# FILE SYSTEM & PATH HELPERS
# ==========================================


def get_descriptors_directory(bovw: BOVW, base_dir: str = "./computed_descriptors") -> str:
    """
    Constructs a unique folder name based on detector config.
    """

    if bovw.dense and bovw.detector_type == "SIFT":
        config_name = f"{bovw.detector_type}_dense_step{bovw.step_size}_size{bovw.scale}"
    else:
        config_name = f"{bovw.detector_type}_keypoints"
    
    params_suffix = ""
    if hasattr(bovw, 'detector_kwargs') and bovw.detector_kwargs:
        if "nfeatures" in bovw.detector_kwargs:
            params_suffix += f"_N{bovw.detector_kwargs['nfeatures']}"
        if "threshold" in bovw.detector_kwargs:
            params_suffix += f"_T{bovw.detector_kwargs['threshold']}"

    full_name = config_name + params_suffix
    return os.path.join(base_dir, full_name)


def check_descriptors_exist(bovw: BOVW, base_dir: str = "./computed_descriptors") -> bool:
    """
    Checks if the descriptors for the given configuration already exist.
    """
    # Check both train and test splits
    for split in ["train", "test"]:
        specific_dir = os.path.join(get_descriptors_directory(bovw, base_dir), split)
        labels_path = os.path.join(specific_dir, "labels.pkl")
        
        # If labels file doesn't exist, we assume descriptors need to be computed
        if not os.path.exists(labels_path):
            return False
            
    return True


# ==========================================
# DATASET LOADING
# ==========================================


def Dataset(ImageFolder:str = "data/MIT_split/train", max_samples: int = None) -> List[Tuple[Type[Image.Image], int]]:

    """
    Loads images from folder structure: ImageFolder/<cls label>/xxx.jpg
    Returns list of (PIL Image, label_index).
    """

    if not os.path.exists(ImageFolder):
        print(f"Error: Folder {ImageFolder} does not exist.")
        return []

    map_classes = {clsi: idx for idx, clsi  in enumerate(os.listdir(ImageFolder))}
    dataset :List[Tuple] = []
    total_images_loaded = 0

    # Iteration for each class folder
    for idx, cls_folder in enumerate(os.listdir(ImageFolder)):
        images_per_class = 0
        image_path = os.path.join(ImageFolder, cls_folder)
        images = glob.glob(image_path + "/*.jpg")

        # Add to dataset every image of the class folder
        for img in images:
            if max_samples is not None and images_per_class >= max_samples:
                break
            try:
                img_pil = Image.open(img).convert("RGB")
                dataset.append((img_pil, map_classes[cls_folder]))
                images_per_class += 1
            except Exception as e:
                print(f"Error loading image {img}: {e}")

        total_images_loaded += images_per_class

    print(f"Loaded {len(dataset)} images from {ImageFolder}")
    return dataset


# ==========================================
# FEATURE EXTRACTION & PROCESSING
# ==========================================


def process_dataset(dataset: List[Tuple[Type[Image.Image], int]], bovw: BOVW, split_name: str, load_from_disk: bool = True,
                    descriptors_root: str = "./computed_descriptors") -> Tuple[List[np.ndarray], List[int]]:
    """
    Extracts features and/or loads them from disk.
    """
    
    specific_dir = os.path.join(get_descriptors_directory(bovw, descriptors_root), split_name)
    os.makedirs(specific_dir, exist_ok=True)
    
    labels_path = os.path.join(specific_dir, "labels.pkl")
    
    all_labels = []

    # Logic: If we are not loading from disk, we compute. 
    if not load_from_disk:
        print(f"[{split_name}] Computing descriptors... Saving to {specific_dir}")
        
        # Compute and Save
        for idx in tqdm.tqdm(range(len(dataset)), desc=f"Phase [{split_name}]: Extracting features"):
            image, label = dataset[idx]
            
            # Extract
            keypoints, descriptors = bovw._extract_features(image=np.array(image))
            
            # Save individual descriptor
            desc_path = os.path.join(specific_dir, f"desc_{idx}.pkl")
            
            if descriptors is None:
                descriptors = np.array([]) 
                keypoints = np.array([])
            else:
                kp_positions = np.array([kp.pt for kp in keypoints]) # Extract coordinates of keypoints

            data_to_save ={
                "descriptors": descriptors,
                "keypoints": kp_positions,
                "image_shape": image.size # 256x256
            }
            
            with open(desc_path, 'wb') as f:
                pickle.dump(data_to_save, f)
            
            # Append label
            all_labels.append(label)

            del descriptors, image, data_to_save
            if idx % 50 == 0: 
                gc.collect()
            
        # Save labels at end
        with open(labels_path, 'wb') as f:
            pickle.dump(all_labels, f)
            
        print(f"[{split_name}] Extraction complete. Descriptors saved to disk.")


    # Loading Process
    print(f"[{split_name}] Gathering file paths...")
    
    # Load Labels
    with open(labels_path, 'rb') as f:
        all_labels = pickle.load(f)
        
    # Load Descriptors
    # Sort based on index in filename: desc_0.pkl, desc_1.pkl...
    desc_paths = glob.glob(os.path.join(specific_dir, "desc_*.pkl"))
    sorted_paths = sorted(desc_paths, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    if len(sorted_paths) != len(all_labels):
        raise ValueError(f"Mismatch! {len(sorted_paths)} files vs {len(all_labels)} labels.")
        
    return sorted_paths, all_labels


def fit_codebook_batched(bovw: BOVW, descriptor_paths: List[str], batch_size: int = 500):
    """
    Loads descriptors in small batches to fit the KMeans codebook without crashing RAM.
    """
    valid_batch = []
    
    print(f"Fitting Codebook in batches of {batch_size}...")
    
    for i, path in enumerate(tqdm.tqdm(descriptor_paths, desc="Fitting Codebook")):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, dict):
                desc = data["descriptors"]
            else:
                desc = data

            # Only add if it has descriptors
            if desc is not None and len(desc) > 0:
                valid_batch.append(desc)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        # If batch is full, update codebook and clear memory
        if len(valid_batch) >= batch_size:
            bovw._update_fit_codebook(descriptors=valid_batch)
            valid_batch = []
            gc.collect()

    # Process remaining items in the last batch
    if len(valid_batch) > 0:
        bovw._update_fit_codebook(descriptors=valid_batch)
        del valid_batch
        gc.collect()


def fit_pca_batched(bovw: BOVW, descriptor_paths: List[str], batch_size: int = 500):
    """
    Loads descriptors in small batches to fit the PCA without crashing RAM.
    """
    if not bovw.use_pca:
        return

    valid_batch = []
    print(f"Fitting PCA in batches of {batch_size}...")
    
    for i, path in enumerate(tqdm.tqdm(descriptor_paths, desc="Fitting PCA")):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, dict):
                desc = data["descriptors"]
            else:
                desc = data

            # Only add if it has descriptors
            if desc is not None and len(desc) > 0:
                valid_batch.append(desc)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        # If batch is full, update pca and clear memory
        if len(valid_batch) >= batch_size:
            bovw.fit_pca_partial(descriptors=valid_batch)
            valid_batch = []
            gc.collect()

    # Process remaining items in the last batch
    if len(valid_batch) > 0:
        bovw.fit_pca_partial(descriptors=valid_batch)
        del valid_batch
        gc.collect()


def extract_bovw_histograms(bovw: Type[BOVW], descriptor_paths: Literal["N", "T", "d"], spatial_pyramid = False):
    """
    Loads one descriptor file at a time, computes the histogram, and discards raw data.
    """
    histograms = []
    
    for path in tqdm.tqdm(descriptor_paths, desc="Computing BoVW Histograms"):
        # Load ONE file
        with open(path, 'rb') as f:
            data = pickle.load(f)

        desc = data["descriptors"]
        keypoints = data["keypoints"]
        shape = data["image_shape"]
        
        # Compute histogram 
        if not spatial_pyramid:
            hist = bovw._compute_codebook_descriptor(
                descriptors=desc, 
                kmeans=bovw.codebook_algo
            )
        else:
            hist = bovw._compute_spatial_pyramid_descriptor(
                descriptors=desc, 
                keypoints=keypoints, 
                image_shape=shape, 
                kmeans=bovw.codebook_algo
            )
        
        histograms.append(hist)

        # Delete raw descriptor immediately
        del data, desc, keypoints
    
    return np.array(histograms)


# ==========================================
# CLASSIFIERS & KERNELS
# ==========================================


def cross_validation(classifier: Type[object], X, y, cv=5):
    """
    Performs cross-validation and prints results.
    """
    print(f"--- Performing {cv}-Fold Cross-Validation ---")
    # skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy') 
    
    print(f"CV Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f}")
    print("---------------------------------------------")
    
    if wandb.run is not None:
        wandb.log({"cv_mean_accuracy": scores.mean()})
    
    return scores.mean()


def histogram_intersection_kernel(X, Y):
    """
    Compute the histogram intersection kernel between X and Y.
    K(x, y) = sum(min(xi, yi))
    """
    # Not the most efficient way, but it does not break the RAM
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        K[i, :] = np.sum(np.minimum(X[i], Y), axis=1)
    
    return K


def get_classifier(name, config):
    seed = config.get("seed", 42)
    common_args = {"class_weight": "balanced", "random_state": seed}
    
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, **common_args, C=config.get("C", 1.0))
    
    # SVM Args
    svm_args = {**common_args, "probability": True, "C": config.get("C", 1.0)}
    if config.get("gamma") is not None:
        svm_args["gamma"] = config["gamma"]

    if name == "SVM-Linear":
        return SVC(kernel='linear', **svm_args)
    elif name == "SVM-RBF":
        return SVC(kernel='rbf', **svm_args)
    
    if name == "SVM-HistogramIntersection":
        return SVC(kernel=histogram_intersection_kernel, **svm_args)
    else:
        raise ValueError(f"Unknown classifier: {name}")


def get_scaler(name: str):
    if name == "StandardScaler":
        return StandardScaler()
    elif name == "MinMaxScaler":
        return MinMaxScaler()
    return None
    

# ==========================================
# TRAIN & TEST PIPELINES
# ==========================================


def train(dataset: List[Tuple], bovw: BOVW, config: dict, load_descriptors: bool = True):
    
    # Get Descriptors
    print("Processing Train Data...")
    train_paths, train_labels = process_dataset(dataset, bovw, split_name="train", load_from_disk=load_descriptors)
    
    # Fit codebook by batches (RAM saving)
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
    scores = cross_val_score(classifier, bovw_histograms, train_labels, cv=5, n_jobs=-1)
    print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    classifier.fit(bovw_histograms, train_labels)

    train_acc = accuracy_score(y_true=train_labels, y_pred=classifier.predict(bovw_histograms))
    print(f"Accuracy on Phase [Train]: {train_acc:.4f}")
    
    if wandb.run is not None:
        wandb.log({"train_accuracy": train_acc})

    return bovw, classifier, scaler, train_acc, scores

    
def test(dataset: List[Tuple], bovw: BOVW, classifier: object, scaler: object, config: dict, load_descriptors: bool = True):
    
    print("Processing Test Data...")
    test_paths, test_labels = process_dataset(dataset, bovw, split_name="test", load_from_disk=load_descriptors)
    
    print("Computing BoVW histograms (Test)...")
    bovw_histograms = extract_bovw_histograms(bovw, test_paths, spatial_pyramid=config["spatial_pyramid"])
    
    if scaler:
        print("Applying Scaler transform...")
        bovw_histograms = scaler.transform(bovw_histograms)
    
    print("Predicting values...")
    y_pred = classifier.predict(bovw_histograms)
    
    test_acc = accuracy_score(y_true=test_labels, y_pred=y_pred)
    print(f"Accuracy on Phase [Test]: {test_acc:.4f}")
    
    if wandb.run is not None:
        wandb.log({"test_accuracy": test_acc})

    return test_acc


def run_experiment(config: dict):
    """
    Orchestrates the data loading, setup, training, and testing based on config.
    """
    
    # --- Pipeline Execution ---
    DATA_PATH = config.get("data_path", "../data/MIT_split/")
    data_train = Dataset(os.path.join(DATA_PATH, "train"), config.get("max_samples_train"))
    data_test = Dataset(os.path.join(DATA_PATH, "test"), config.get("max_samples_test"))
    
    if not data_train or not data_test:
        raise ValueError("Dataset empty.")

    # Initialize BoVW
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

    # Check Descriptors existence to decide on compute vs load
    load_descriptors = False
    if check_descriptors_exist(bovw):
        print("Descriptors found on disk. Loading...")
        load_descriptors = True
    else:
        print("Descriptors not found. Computing...")

    # Train
    bovw, classifier, scaler, train_acc, cv_scores = train(data_train, bovw, config, load_descriptors=load_descriptors)

    # Test
    test_acc = test(data_test, bovw, classifier, scaler, config, load_descriptors=load_descriptors)

    return train_acc, test_acc, cv_scores


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    
    # --- CONFIGURATION DICTIONARY ---
    CONFIG = {
        "data_path": "../data/MIT_split/",
        "seed": 42,
        
        # Features
        "detector": "SIFT",          # SIFT, ORB, AKAZE
        "detector_kwargs": {"nfeatures": 500},
        
        "dense": False,
        "step_size": 8,              # Only for Dense
        "scale": 8,                  # Only for Dense
        
        # BoVW
        "codebook_size": 1024,
        "spatial_pyramid": False,
        "levels": [1, 2],
        
        "use_pca": False,
        "pca_components": 64,

        "normalization": "l2", # l1, l2, none
        "scaler": None,        # StandardScaler, MinMaxScaler, None

        # Classifier
        "classifier": "SVM-Linear",  # LogisticRegression, SVM-Linear, SVM-RBF
        "C": 1.0,
        "gamma": "scale",

        "codebook_batch_size": 10000,
        
        # Debug
        "max_samples_train": None,
        "max_samples_test": None
    }

    # Generate Run Name
    run_name = f"{CONFIG['detector']}_dense{CONFIG['dense']}_k{CONFIG['codebook_size']}_{CONFIG['classifier']}"
    if CONFIG['dense']: run_name += f"_step{CONFIG['step_size']}"
    
    # Initialize WandB
    if CONFIG["max_samples_train"] is None:
        wandb.init(project="project-4-week1", name=run_name, config=CONFIG)
    else:
        print("Debug mode: WandB disabled.")

    # RUN
    start_time = time.time()
    run_experiment(CONFIG)
    print(f"Done in {time.time() - start_time:.2f}s")