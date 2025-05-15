import streamlit as st
import numpy as np
import cv2
import joblib
import mediapipe as mp
from PIL import Image

from features.hog import extract_hog_features
from features.lbp import extract_lbp_features
from features.sobel import extract_sobel_features
from features.landmarks import extract_landmark_features



model = joblib.load("F:/UIT HK4/Introduction to Computer Vision CS231/FER_Demo/FER/model/combined_model.pkl")
classes = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']

st.title("Facial Expression Recognition 🎭")
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(img_rgb)

        if not results.detections:
            st.error("No face detected. Please upload a clearer image.")
        else:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = x1 + int(bbox.width * w)
            y2 = y1 + int(bbox.height * h)

            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)

            face_img = img[y1:y2, x1:x2]
            face_img_resized = cv2.resize(face_img, (100, 100)) 

            face_pil = Image.fromarray(cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2RGB))

            gray_img = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2GRAY)

            st.subheader("Feature Extraction...")

            # Extract features
            hog_feat = extract_hog_features([gray_img])
            lbp_feat = extract_lbp_features([gray_img])
            sobel_feat = extract_sobel_features([gray_img])

            # Giảm chiều Sobel bằng cách cắt
            n_components_sobel = 200
            if sobel_feat.shape[1] >= n_components_sobel:
                sobel_feat_reduced = sobel_feat[:, :n_components_sobel]
            else:
                padding = np.zeros((sobel_feat.shape[0], n_components_sobel - sobel_feat.shape[1]))
                sobel_feat_reduced = np.hstack((sobel_feat, padding))

            landmark_feat = extract_landmark_features([gray_img])
            X_combined = np.concatenate([hog_feat, lbp_feat, sobel_feat_reduced, landmark_feat], axis=1)

            # Debug shape
            st.write(f"Combined features shape: {X_combined.shape}")

# Prediction
            pred = model.predict(X_combined)[0]
            prob = model.predict_proba(X_combined)[0]

            col1, col2 = st.columns([1.5, 1], gap="large")  # Tăng khoảng cách giữa 2 cột

            with col1:
                face_for_model = face_pil.copy()

# Resize ảnh chỉ để hiển thị to hơn (không ảnh hưởng tới mô hình)
                display_face = face_pil.resize((100, 100))  

                # Hiển thị trong UI
                st.image(display_face, caption="Detected Face",use_container_width=True)

            with col2:
                st.markdown(
                    """
                    <style>
                    .emotion-header {
                        font-size: 24px;
                        font-weight: bold;
                        margin-bottom: 10px;
                    }
                    .emotion-section {
                        font-size: 18px;
                        line-height: 1.8;
                        padding-left: 10px;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown('<div class="emotion-header">Prediction</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="emotion-section"><b>Predicted Expression:</b> {classes[pred]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)



