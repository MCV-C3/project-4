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

import os
# Suppress joblib warning about physical cores
os.environ['LOKY_MAX_CPU_COUNT'] = "4"

from bovw import BOVW
import wandb

from typing import *
from PIL import Image

import numpy as np
import pickle
import glob
import tqdm
import os
import gc

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics.pairwise import additive_chi2_kernel
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score


def get_descriptors_directory(bovw: BOVW, base_dir: str = "./computed_descriptors") -> str:
    """
    Constructs a unique folder name based on detector config.
    """
    if bovw.dense and bovw.detector_type == "SIFT":
        config_name = f"{bovw.detector_type}_dense_step{bovw.step_size}_size{bovw.keypoint_size}"
    else:
        config_name = f"{bovw.detector_type}_keypoints"
    
    return os.path.join(base_dir, config_name)


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


def process_dataset(dataset: List[Tuple[Type[Image.Image], int]], 
                    bovw: BOVW, 
                    split_name: str,
                    load_from_disk: bool = True,
                    descriptors_root: str = "./computed_descriptors") -> Tuple[List[np.ndarray], List[int]]:
    """
    Extracts features and/or loads them from disk.
    """
    
    specific_dir = os.path.join(get_descriptors_directory(bovw, descriptors_root), split_name)
    os.makedirs(specific_dir, exist_ok=True)
    
    labels_path = os.path.join(specific_dir, "labels.pkl")
    
    # all_descriptors = []
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

            del descriptors
            del image
            if idx % 50 == 0: 
                gc.collect()
            
        # Save labels at end
        with open(labels_path, 'wb') as f:
            pickle.dump(all_labels, f)
            
        # Clear dataset from memory if possible
        # gc.collect() 
        print(f"[{split_name}] Extraction complete. Descriptors saved to disk.")


    # Load Process
    print(f"[{split_name}] Gathering file paths...")
    # print(f"[{split_name}] Loading descriptors from disk: {specific_dir}")
    
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


def fit_codebook_batched(bovw: BOVW, descriptor_paths: List[str], batch_size: int = 100):
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
            valid_batch = [] # Reset list
            gc.collect()     # Force memory release

    # Process remaining items in the last batch
    if len(valid_batch) > 0:
        bovw._update_fit_codebook(descriptors=valid_batch)
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
        
        # Compute histogram (returns a small 1D array, e.g., size 1024)
        if not spatial_pyramid:
            hist = bovw._compute_codebook_descriptor(descriptors=desc, kmeans=bovw.codebook_algo)
        else:
            hist = bovw._compute_spatial_pyramid_descriptor(descriptors=desc, keypoints=keypoints, 
                                                            image_shape=shape, kmeans=bovw.codebook_algo)
    
        histograms.append(hist)

        # Delete raw descriptor immediately
        del data, desc, keypoints
    
    return np.array(histograms)


def cross_validation(classifier: Type[object], X, y, cv=5):
    """
    Performs cross-validation and prints results.
    """
    print(f"--- Performing {cv}-Fold Cross-Validation ---")
    # skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy', verbose=1, n_jobs=1) 
    
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


def get_classifier(classifier_name: str, **kwargs) -> object:
    """
    Factory to get the classifier based on name.
    """
    if classifier_name == "LogisticRegression":
        return LogisticRegression(class_weight="balanced", max_iter=1000, **kwargs)
    elif classifier_name == "SVM-Linear":
        return SVC(kernel='linear', class_weight='balanced', probability=True, **kwargs)
    elif classifier_name == "SVM-RBF":
        return SVC(kernel='rbf', class_weight='balanced', probability=True, **kwargs)
    elif classifier_name == "SVM-HistogramIntersection":
        return SVC(kernel=histogram_intersection_kernel, class_weight='balanced', probability=True, **kwargs)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")



def train(dataset: List[Tuple[Type[Image.Image], int]], bovw: Type[BOVW], load_descriptors: bool = True, spatial_pyramid: bool = False, classifier_kwargs: dict = {}):
    
    # Get Descriptors
    print("Processing Train Data...")
    train_paths, train_labels = process_dataset(dataset, bovw, split_name="train", load_from_disk=load_descriptors)
    
    # --- PCA Training ---
    if bovw.pca is not None:
        print("Training PCA...")
        # Load a random subset of descriptors for PCA
        # We need enough samples. standard PCA needs all in memory.
        # Let's load ~20k descriptors max (or more depending on RAM).
        # Actually, let's just pick N random files and load them.
        
        pca_samples = []
        max_pca_samples = 50000 
        current_samples = 0
        
        import random
        random.seed(42)
        # Shuffle paths to get random subset
        pca_paths = train_paths[:]
        random.shuffle(pca_paths)
        
        for path in tqdm.tqdm(pca_paths, desc="Loading descriptors for PCA"):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            desc = data["descriptors"] if isinstance(data, dict) else data
            
            if desc is not None and len(desc) > 0:
                pca_samples.append(desc)
                current_samples += len(desc)
            
            if current_samples >= max_pca_samples:
                break
        
        bovw.fit_pca(pca_samples)
        # Release memory
        del pca_samples
        gc.collect()
    # --------------------

    # Fit codebook by batches (RAM saving)
    fit_codebook_batched(bovw, train_paths, batch_size=500)
    # bovw._update_fit_codebook(descriptors=valid_descriptors_for_fitting)

    print("Computing BoVW histograms [Train]...")
    bovw_histograms = extract_bovw_histograms(bovw=bovw, descriptor_paths=train_paths, spatial_pyramid=spatial_pyramid) 
    
    print(f"Fitting the classifier: {bovw.classifier_name}...")
    classifier = get_classifier(bovw.classifier_name, **classifier_kwargs)

    # Cross Validation
    cv_score = cross_validation(classifier, bovw_histograms, train_labels, cv=5)

    classifier.fit(bovw_histograms, train_labels)

    train_acc = accuracy_score(y_true=train_labels, y_pred=classifier.predict(bovw_histograms))
    print(f"Accuracy on Phase [Train]: {train_acc:.4f}")
    
    if wandb.run is not None:
        wandb.log({"train_accuracy": train_acc})

    return bovw, classifier, train_acc, cv_score

    
def test(dataset: List[Tuple[Type[Image.Image], int]], bovw: Type[BOVW], classifier: Type[object], load_descriptors: bool = True, spatial_pyramid: bool = False):
    
    print("Processing Test Data...")
    test_paths, test_labels = process_dataset(dataset, bovw, split_name="test", load_from_disk=load_descriptors)
    
    print("Computing BoVW histograms (Test)...")
    bovw_histograms = extract_bovw_histograms(bovw=bovw, descriptor_paths=test_paths, spatial_pyramid=spatial_pyramid)
    
    print("Predicting values...")
    y_pred = classifier.predict(bovw_histograms)
    
    test_acc = accuracy_score(y_true=test_labels, y_pred=y_pred)
    print(f"Accuracy on Phase [Test]: {test_acc:.4f}")
    
    if wandb.run is not None:
        wandb.log({"test_accuracy": test_acc})


def Dataset(ImageFolder:str = "data/MIT_split/train", max_samples: int = None) -> List[Tuple[Type[Image.Image], int]]:

    """
    Simple dataset loader
        
    Expected Structure:
        ImageFolder/<cls label>/xxx1.jpg
        ImageFolder/<cls label>/xxx2.jpg
        ...

    Example:
        ImageFolder/cat/123.jpg
        ImageFolder/cat/nsdf3.jpg
        ...
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


def run_experiment(
    detector: str = "SIFT",
    dense: bool = False,
    codebook_size: int = 1024,
    spatial_pyramid: bool = False,
    levels: List[int] = [1, 2],
    step_size: int = 8,
    keypoint_size: int = 8,
    classifier: str = "SVM-Linear",
    c_param: float = 1.0,
    gamma_param: Union[str, float] = 'scale',
    pca_components: Optional[int] = None,
    max_samples_train: Optional[int] = None,
    max_samples_test: Optional[int] = None,
    data_path: str = "../data/MIT_split/",
    wandb_project: str = "project-4-week1",
    use_wandb: bool = True
) -> Dict[str, Any]:

    # --- Auto-Detect Descriptors ---
    # Create a temporary BOVW instance to check paths (parameters must match)
    temp_bovw = BOVW(
        detector_type=detector,
        dense=dense,
        step_size=step_size,
        keypoint_size=keypoint_size,
        codebook_size=codebook_size
    )
    
    if check_descriptors_exist(temp_bovw):
        print("Found existing descriptors. Loading from disk.")
        load_descriptors = True
    else:
        print("Descriptors not found. Computing them.")
        load_descriptors = False

    # ---------------------
    
    # Construct descriptive run name
    run_name = f"{detector}_dense{dense}_k{codebook_size}_pyramid{spatial_pyramid}_{classifier}_C{c_param}"
    if classifier == "SVM-RBF":
        run_name += f"_gamma{gamma_param}"
        
    if dense:
        run_name += f"_step{step_size}_size{keypoint_size}"
        
    if pca_components is not None:
        run_name += f"_pca{pca_components}"

    # Verify data path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    # Initialize WandB
    if use_wandb:
        # Re-init is important for sweeps
        wandb.init(
            project=wandb_project,
            name=run_name,
            reinit=True, 
            config={
                "detector": detector,
                "spatial_pyramid": spatial_pyramid,
                "levels": levels,
                "dense": dense,
                "step_size": step_size,
                "keypoint_size": keypoint_size,
                "max_samples_train": max_samples_train,
                "max_samples_test": max_samples_test,
                "codebook_size": codebook_size,
                "classifier": classifier,
                "C": c_param,
                "gamma": gamma_param,
                "pca_components": pca_components
            }
        )
    else:
        print("WandB logging is DISABLED.")

    print("Loading Dataset...")
    data_train = Dataset(ImageFolder=os.path.join(data_path, "train"), max_samples=max_samples_train)
    data_test = Dataset(ImageFolder=os.path.join(data_path, "test"), max_samples=max_samples_test) 
    
    if not data_train or not data_test:
        if use_wandb: wandb.finish()
        raise ValueError("Dataset empty. Check paths.")
    
    # Initialize BoVW
    bovw = BOVW(
        detector_type=detector,
        dense=dense,
        step_size=step_size,
        keypoint_size=keypoint_size,
        codebook_size=codebook_size,
        levels=levels,
        pca_components=pca_components
    )
    
    # Attach classifier name to BOVW object for convenience
    bovw.classifier_name = classifier

    import time
    start_time = time.time()
    
    # Prepare classifier kwargs
    classifier_kwargs = {"C": c_param}
    if classifier == "SVM-RBF":
        classifier_kwargs["gamma"] = gamma_param

    # Train
    bovw, classifier_obj, train_acc, cv_score = train(dataset=data_train, bovw=bovw, load_descriptors=load_descriptors, spatial_pyramid=spatial_pyramid, classifier_kwargs=classifier_kwargs)
    
    # Calculate Train Accuracy explicitly if needed mostly it is printed inside train but good to return
    # We can assume train() prints it. The object classifier is fitted.
    
    # Test
    # We need to capture accurate metrics. The test function acts as a procedure, let's copy the evaluation logic here or modify test to return values.
    # For minimal changes, I'll repeat the evaluation part or rely on what 'test' prints/logs? 
    # Better to have 'test' return accuracy or just compute it here.
    # The 'test' function computes and logs it. Let's make 'test' return the accuracy too in a separate small edit or just copy the logic.
    # Actually, I'll just copy the evaluation logic here to be safe and have return values.
    
    print("Processing Test Data...")
    test_paths, test_labels = process_dataset(data_test, bovw, split_name="test", load_from_disk=load_descriptors)
    
    print("Computing BoVW histograms (Test)...")
    bovw_histograms_test = extract_bovw_histograms(bovw=bovw, descriptor_paths=test_paths, spatial_pyramid=spatial_pyramid)
    
    print("Predicting values...")
    y_pred = classifier_obj.predict(bovw_histograms_test)
    
    test_acc = accuracy_score(y_true=test_labels, y_pred=y_pred)
    print(f"Accuracy on Phase [Test]: {test_acc:.4f}")

    end_time = time.time()
    duration = end_time - start_time
    
    if use_wandb:
        wandb.log({
            "test_accuracy": test_acc,
            "execution_time_seconds": duration, 
            "execution_time_minutes": duration/60
        })
        wandb.finish() # Finish the run so next one can start

    return {
        "detector": detector,
        "dense": dense,
        "classifier": classifier,
        "C": c_param,
        "gamma": gamma_param,
        "test_accuracy": test_acc,
        "train_accuracy": train_acc,
        "cv_accuracy": cv_score,
        "time": duration
    }


if __name__ == "__main__":
    # Default behavior if run directly
    run_experiment(
        detector="SIFT",
        dense=False,
        codebook_size=1024,
        classifier="SVM-Linear",
        c_param=1.0,
        use_wandb=True,
        # max_samples_train=100, # Uncomment for testing
        # max_samples_test=10
    )
