import cv2
import dlib
import numpy as np
import urllib.request
from matplotlib import pyplot as plt

frontalface_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(r'F:\UIT HK4\Introduction to Computer Vision CS231\FER_Demo\shape_predictor_68_face_landmarks.dat')


from scipy.spatial import distance
def extract_landmark_features(images, predictor_path = r"F:\UIT HK4\Introduction to Computer Vision CS231\FER_Demo\shape_predictor_68_face_landmarks.dat"):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    all_features = []

    for image in images:
        faces = detector(image)
        if len(faces) > 0:
            # Select the first detected face
            shape = predictor(image, faces[0])
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])

            # Raw coordinates
            raw_features = landmarks.flatten()

            # Pairwise distances between selected landmark pairs
            # Example: distance between eyes (points 36 and 45)
            eye_distance = distance.euclidean(landmarks[36], landmarks[45])

            # Eye Aspect Ratio (EAR)
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            left_EAR = compute_ear(left_eye)
            right_EAR = compute_ear(right_eye)

            # Mouth Aspect Ratio (MAR)
            mouth = landmarks[60:68]
            MAR = compute_mar(mouth)

            # Combine all features
            features = np.concatenate([raw_features, [eye_distance, left_EAR, right_EAR, MAR]])
        else:
            # If no face is detected, assign zero vector
            features = np.zeros(68 * 2 + 4)

        all_features.append(features)

    return np.array(all_features)

def compute_ear(eye):
    # Compute Eye Aspect Ratio
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def compute_mar(mouth):
    # Compute Mouth Aspect Ratio
    A = distance.euclidean(mouth[1], mouth[7])
    B = distance.euclidean(mouth[2], mouth[6])
    C = distance.euclidean(mouth[3], mouth[5])
    D = distance.euclidean(mouth[0], mouth[4])
    mar = (A + B + C) / (3.0 * D)
    return mar

