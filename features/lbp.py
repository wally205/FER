import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from tqdm.notebook import tqdm

def extract_lbp_features(images, P=8, R=1, grid_size=(6, 6)):
    lbp_features = []
    for img in tqdm(images, desc="Processing data"):
        h, w = img.shape
        cell_h, cell_w = h // grid_size[0], w // grid_size[1]
        feature_vector = []

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell = img[y1:y2, x1:x2]

                lbp_cell = local_binary_pattern(cell, P, R, method='uniform')
                hist, _ = np.histogram(lbp_cell.ravel(),
                                       bins=np.arange(0, P + 3),
                                       range=(0, P + 2))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-6)
                feature_vector.extend(hist)
        lbp_features.append(feature_vector)
    return np.array(lbp_features)