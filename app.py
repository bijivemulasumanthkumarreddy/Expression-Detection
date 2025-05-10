import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('emotion_model.h5')
labels = ['Angry', 'Happy', 'Neutral', 'Sad']

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        if frame is None:
            return None

        img = frame.to_ndarray(format="bgr24")

        # Preprocess the frame for prediction
        face_img = cv2.resize(img, (224, 224))
        face_arr = face_img / 255.0
        face_arr = np.expand_dims(face_arr, axis=0)

        # Predict emotion
        preds = model.predict(face_arr)[0]
        label = labels[np.argmax(preds)]

        # Display label on the frame
        cv2.putText(img, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

st.title("Real-Time Emotion Detection")

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
