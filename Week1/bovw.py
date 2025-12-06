import os
from typing import *

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans


class BOVW():
    
    def __init__(self, detector_type="AKAZE", dense:bool=False, step_size:int=8, scale:int=8, codebook_size:int=50, 
                 levels:list=[1], random_state:int=42, detector_kwargs:dict={}, codebook_kwargs:dict={}):

        if dense and detector_type != 'SIFT':
            raise ValueError("Dense sampling is currently only supported for SIFT.")
        
        if detector_type == 'SIFT':
            self.detector = cv2.SIFT_create(**detector_kwargs)
        elif detector_type == 'AKAZE':
            self.detector = cv2.AKAZE_create(**detector_kwargs)
        elif detector_type == 'ORB':
            self.detector = cv2.ORB_create(**detector_kwargs)
        else:
            raise ValueError("Detector type must be 'SIFT', 'SURF', or 'ORB'")

        self.detector_type = detector_type
        self.detector_kwargs = detector_kwargs

        # Codebook / KMeans parameters
        self.codebook_size = codebook_size
        self.random_state = random_state
        self.codebook_algo = MiniBatchKMeans(
            n_clusters=self.codebook_size, n_init='auto', 
            random_state=random_state, **codebook_kwargs
        )

        # Parameters for Dense SIFT
        self.dense = dense
        self.step_size = step_size
        self.scale = scale

        # Spatial Pyramid levels
        self.levels = levels

               
    ## Modify this function in order to be able to create a dense sift
    def _extract_features(self, image: Literal["H", "W", "C"]) -> Tuple:
        """
        Extracts descriptors from the image. 
        """

        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        if self.dense and self.detector_type == "SIFT":
            # Case for Dense SIFT
            keypoints = []
            h, w = image.shape[:2]
            
            # Creation of keypoints for each determinate step_size
            for y in range(0, h, self.step_size):
                for x in range(0, w, self.step_size):
                    kp = cv2.KeyPoint(float(x), float(y), float(self.scale))
                    keypoints.append(kp)
            
            if not keypoints:
                return (), None
            
            # Compute descriptors for every created keypoints
            keypoints, descriptors = self.detector.compute(image, keypoints)
            return keypoints, descriptors
            
        else:
            # Standard detection and compute of descriptors
            return self.detector.detectAndCompute(image, None)
    
    
    def _update_fit_codebook(self, descriptors: Literal["N", "T", "d"])-> Tuple[Type[MiniBatchKMeans],Literal["codebook_size", "d"]]:
        """
        Partially fits the KMeans algorithm with a batch of descriptors.
        """
        # Filter out None descriptors (images with no features)
        valid_descriptors = [d for d in descriptors if d is not None and len(d) > 0]
        if not valid_descriptors:
            raise ValueError("Error: No valid descriptors found. Cannot fit codebook.")
        
        all_descriptors = np.vstack(descriptors)
        self.codebook_algo = self.codebook_algo.partial_fit(X=all_descriptors)

        return self.codebook_algo, self.codebook_algo.cluster_centers_


    def _compute_codebook_descriptor(self, descriptors: Literal["1 T d"], kmeans: Type[KMeans]) -> np.ndarray:
        """
        Computes a standard BoVW histogram (frequency of visual words).
        """
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(kmeans.n_clusters)
        
        visual_words = kmeans.predict(descriptors)
        
        # Create a histogram of visual words
        codebook_descriptor = np.zeros(kmeans.n_clusters)
        for label in visual_words:
            codebook_descriptor[label] += 1
        
        # Normalize the histogram 
        norm = np.linalg.norm(codebook_descriptor)
        if norm > 0:
            codebook_descriptor /= norm
        
        return codebook_descriptor       
    

    def _compute_spatial_pyramid_descriptor(self, descriptors: np.ndarray, keypoints: np.ndarray, 
                                            image_shape: Tuple[int, int], kmeans: Type[KMeans]) -> np.ndarray:
        """
        Computes the Spatial Pyramid Matching (SPM) histogram.
        Concatenates histograms from different grid levels.
        """
        if descriptors is None or len(descriptors) == 0:
            # Return zero vector of correct size
            total_bins = sum([l*l for l in self.levels]) * self.codebook_size
            return np.zeros(total_bins)

        W, H = image_shape
        
        # Prediction of visual words for ALL descriptors first
        visual_words = kmeans.predict(descriptors)
        
        pyramid_histogram = []

        # Levels iteration
        for level in self.levels:
            n_cols = level
            n_rows = level
            
            w_step = W / n_cols
            h_step = H / n_rows

            # Iterate through the grid cells
            for r in range(n_rows):
                for c in range(n_cols):
                    # Define boundaries of the cell
                    x_min = c * w_step
                    x_max = (c + 1) * w_step
                    y_min = r * h_step
                    y_max = (r + 1) * h_step

                    # Find keypoints inside this cell
                    in_x = (keypoints[:, 0] >= x_min) & (keypoints[:, 0] < x_max)
                    in_y = (keypoints[:, 1] >= y_min) & (keypoints[:, 1] < y_max)
                    in_cell_indices = np.where(in_x & in_y)[0]

                    # Create Histogram for this cell
                    hist = np.zeros(self.codebook_size)
                    
                    if len(in_cell_indices) > 0:
                        cell_words = visual_words[in_cell_indices]
                        for w_idx in cell_words:
                            hist[w_idx] += 1
                    
                    # TODO: Normalization here helps??

                    pyramid_histogram.extend(hist)

        pyramid_histogram = np.array(pyramid_histogram)

        # Normalization of concatenated histogram
        norm = np.linalg.norm(pyramid_histogram)
        if norm > 0:
            pyramid_histogram /= norm
            
        return pyramid_histogram


def visualize_bow_histogram(histogram, image_index, output_folder="./test_example.jpg"):
    """
    Visualizes the Bag of Visual Words histogram for a specific image and saves the plot to the output folder.
    
    Args:
        histogram (np.array): BoVW histogram.
        cluster_centers (np.array): Cluster centers (visual words).
        image_index (int): Index of the image for reference.
        output_folder (str): Folder where the plot will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(histogram)), histogram)
    plt.title(f"BoVW Histogram for Image {image_index}")
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.xticks(range(len(histogram)))
    
    # Save the plot to the output folder
    plot_path = os.path.join(output_folder, f"bovw_histogram_image_{image_index}.png")
    plt.savefig(plot_path)
    
    # Optionally, close the plot to free up memory
    plt.close()

    print(f"Plot saved to: {plot_path}")

