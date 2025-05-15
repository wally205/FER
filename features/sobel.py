import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def extract_sobel_features(images):
    all_features = []
    for image in images:
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel_mag = cv2.normalize(sobel_mag, None, 0, 1.0, cv2.NORM_MINMAX)
        features = sobel_mag.flatten()
        all_features.append(features)

    X_sobel=np.array(all_features)

    return np.array(X_sobel)

