from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np
from tqdm import tqdm


def extract_hog_features(images, resize_shape=(64, 64)):
    hog_features = []
    for img in tqdm(images):
        resized = resize(img, resize_shape)
        features = hog(resized,
                       orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       block_norm='L2-Hys')
        hog_features.append(features)
    return np.array(hog_features)
