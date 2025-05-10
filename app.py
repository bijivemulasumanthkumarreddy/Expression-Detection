
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model('emotion_model.h5')
labels = ['Angry', 'Happy', 'Neutral', 'Sad']

# Webcam input
st.title("Real-Time Emotion Detection")
run = st.checkbox('Start Camera')

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

def is_live_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var > 100

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Webcam not accessible.")
        break

    face_img = cv2.resize(frame, (224, 224))
    face_arr = img_to_array(face_img) / 255.0
    face_arr = np.expand_dims(face_arr, axis=0)

    if is_live_face(frame):
        preds = model.predict(face_arr)[0]
        label = labels[np.argmax(preds)]
    else:
        label = "Spoofing Detected!"

    frame = cv2.putText(frame, label, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
