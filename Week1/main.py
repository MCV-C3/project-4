from bovw import BOVW

from typing import *
from PIL import Image

import numpy as np
import pickle
import glob
import tqdm
import os

from sklearn.linear_model import LogisticRegression
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


def process_dataset(dataset: List[Tuple[Type[Image.Image], int]], 
                    bovw: BOVW, 
                    split_name: str,
                    load_from_disk: bool = True,
                    descriptors_root: str = "./computed_descriptors") -> Tuple[List[np.ndarray], List[int]]:
    """
    Extracts features or loads them from disk.
    """
    
    specific_dir = os.path.join(get_descriptors_directory(bovw, descriptors_root), split_name)
    os.makedirs(specific_dir, exist_ok=True)
    
    labels_path = os.path.join(specific_dir, "labels.pkl")
    
    all_descriptors = []
    all_labels = []

    if load_from_disk:
        print(f"[{split_name}] Loading descriptors from disk: {specific_dir}")
        
        # Load Labels
        with open(labels_path, 'rb') as f:
            all_labels = pickle.load(f)
            
        # Load Descriptors
        # Sort based on index in filename: desc_0.pkl, desc_1.pkl...
        desc_paths = glob.glob(os.path.join(specific_dir, "desc_*.pkl"))
        desc_files = sorted(desc_paths, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        # Verify length match
        if len(desc_files) != len(all_labels):
            raise ValueError(f"Mismatch in files! Found {len(desc_files)} descriptor files but {len(all_labels)} labels.")

        for fpath in tqdm.tqdm(desc_files, desc=f"Loading {split_name}"):
            with open(fpath, 'rb') as f:
                desc = pickle.load(f)
                all_descriptors.append(desc)
                
    else:
        print(f"[{split_name}] Computing descriptors... Saving to {specific_dir}")
        
        # Compute and Save
        for idx in tqdm.tqdm(range(len(dataset)), desc=f"Phase [{split_name}]: Extracting features"):
            image, label = dataset[idx]
            
            # Extract
            _, descriptors = bovw._extract_features(image=np.array(image))
            
            # Save individual file
            desc_path = os.path.join(specific_dir, f"desc_{idx}.pkl")
            
            if descriptors is None:
                descriptors = np.array([]) 
            
            with open(desc_path, 'wb') as f:
                pickle.dump(descriptors, f)
                
            all_descriptors.append(descriptors)
            all_labels.append(label)
            
        # Save labels at end
        with open(labels_path, 'wb') as f:
            pickle.dump(all_labels, f)
            
    return all_descriptors, all_labels


def extract_bovw_histograms(bovw: Type[BOVW], descriptors: Literal["N", "T", "d"]):
    """
    Converts list of raw descriptors into list of BoVW histograms.
    """
    return np.array([bovw._compute_codebook_descriptor(descriptors=descriptor, kmeans=bovw.codebook_algo) for descriptor in descriptors])


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
    return scores.mean()


def train(dataset: List[Tuple[Type[Image.Image], int]], bovw: Type[BOVW], load_descriptors: bool = True):
    
    # Get Descriptors
    print("Processing Train Data...")
    train_descriptors, train_labels = process_dataset(dataset, bovw, split_name="train", load_from_disk=load_descriptors)
    
    # Filter valid descriptors for fitting Codebook
    valid_descriptors_for_fitting = [d for d in train_descriptors if d is not None and len(d) > 0]
    
    if len(valid_descriptors_for_fitting) == 0:
        raise ValueError("No valid descriptors found in training set to fit the codebook.")
            
    print(f"Fitting codebook with {len(valid_descriptors_for_fitting)} valid images...")
    bovw._update_fit_codebook(descriptors=valid_descriptors_for_fitting)

    print("Computing BoVW histograms [Train]...")
    bovw_histograms = extract_bovw_histograms(bovw=bovw, descriptors=train_descriptors) 
    
    print("Fitting the classifier...")
    classifier = LogisticRegression(class_weight="balanced", max_iter=1000)

    # Cross Validation
    cross_validation(classifier, bovw_histograms, train_labels, cv=5)

    classifier.fit(bovw_histograms, train_labels)

    train_acc = accuracy_score(y_true=train_labels, y_pred=classifier.predict(bovw_histograms))
    print(f"Accuracy on Phase [Train]: {train_acc:.4f}")

    return bovw, classifier

    
def test(dataset: List[Tuple[Type[Image.Image], int]], bovw: Type[BOVW], classifier: Type[object], load_descriptors: bool = True):
    
    print("Processing Test Data...")
    test_descriptors, test_labels = process_dataset(dataset, bovw, split_name="test", load_from_disk=load_descriptors)
    
    print("Computing BoVW histograms (Test)...")
    bovw_histograms = extract_bovw_histograms(bovw=bovw, descriptors=test_descriptors)
    
    print("Predicting values...")
    y_pred = classifier.predict(bovw_histograms)
    
    test_acc = accuracy_score(y_true=test_labels, y_pred=y_pred)
    print(f"Accuracy on Phase [Test]: {test_acc:.4f}")


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


if __name__ == "__main__":

    DATA_PATH = "../data/MIT_split/"

    # --- Configuration ---
    DETECTOR = "ORB"  # SIFT, ORB, AKAZE

    DENSE = False
    STEP_SIZE = 8
    KEYPOINT_SIZE = 8 
    
    # Subsampling for algorithm testing (set to None for full run)
    MAX_SAMPLES_TRAIN = 100 # Quantity of images per label !!!! Not in total
    MAX_SAMPLES_TEST = 10 # Same as train samples
    
    # TODO: Change this in order to automatize the load from disk or not
    LOAD_DESCRIPTORS = True 

    # ---------------------

    print("Loading Dataset...")
    data_train = Dataset(ImageFolder=DATA_PATH + "train", max_samples=None)
    data_test = Dataset(ImageFolder=DATA_PATH + "test", max_samples=None) 

    if not data_train or not data_test:
        raise ValueError("Dataset empty. Check paths.")
    
    # Initialize BoVW
    bovw = BOVW(
        detector_type=DETECTOR,
        dense=DENSE,
        step_size=STEP_SIZE,
        keypoint_size=KEYPOINT_SIZE,
        codebook_size=1024
    )
    
    # Train
    bovw, classifier = train(dataset=data_train, bovw=bovw, load_descriptors=LOAD_DESCRIPTORS)
    
    # Test
    test(dataset=data_test, bovw=bovw, classifier=classifier, load_descriptors=LOAD_DESCRIPTORS)