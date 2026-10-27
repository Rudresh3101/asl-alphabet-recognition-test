import json
import threading
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "asl_mobilenetv2.keras"
LABELS_PATH = BASE_DIR / "models" / "class_names.json"
IMAGE_SIZE = 160
CONFIDENCE_THRESHOLD = 0.50

st.set_page_config(page_title="ASL Alphabet Recognition", page_icon="🤟", layout="centered")

@st.cache_resource
def load_asl_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if isinstance(labels, dict):
        labels = [name for name, index in sorted(labels.items(), key=lambda item: int(item[1]))]
    labels = [str(x) for x in labels]
    if len(labels) != model.output_shape[-1]:
        raise ValueError(f"Model has {model.output_shape[-1]} outputs but {len(labels)} labels were found.")
    return model, labels

# The model expects RGB pixels in the original 0-255 range, resized to 160x160.
def predict_image(image, model, labels):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)[None, ...]
    probabilities = model.predict(array, verbose=0)[0]
    top_indices = np.argsort(probabilities)[-3:][::-1]
    best_index = int(top_indices[0])
    return labels[best_index], float(probabilities[best_index]), [
        (labels[int(i)], float(probabilities[int(i)])) for i in top_indices
    ]

st.title("🤟 ASL Alphabet Recognition")
st.write("Recognize static American Sign Language (ASL) alphabet signs using a trained MobileNetV2 model.")
st.info("Show one clear hand sign, keep the whole hand in frame, use good lighting, and keep the background simple.")

try:
    model, class_names = load_asl_model()
except Exception as exc:
    st.error("The model could not be loaded. Check that the model files are present.")
    st.exception(exc)
    st.stop()

mode = st.radio("Choose prediction mode", ["Static image", "Real-time webcam"], horizontal=True)

if mode == "Static image":
    source = st.radio("Image source", ["Upload an image", "Capture a photo"], horizontal=True)
    image = None
    if source == "Upload an image":
        uploaded_file = st.file_uploader("Upload a hand-sign image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            try:
                image = ImageOps.exif_transpose(Image.open(uploaded_file))
            except Exception as exc:
                st.error(f"Could not open this image: {exc}")
    else:
        camera_file = st.camera_input("Capture your ASL hand sign")
        if camera_file is not None:
            try:
                image = ImageOps.exif_transpose(Image.open(camera_file))
            except Exception as exc:
                st.error(f"Could not open the camera image: {exc}")

    if image is not None:
        st.image(image, caption="Image submitted for prediction", use_container_width=True)
        if st.button("Recognize sign", type="primary", use_container_width=True):
            with st.spinner("Recognizing sign..."):
                label, confidence, top_predictions = predict_image(image, model, class_names)
            if confidence >= CONFIDENCE_THRESHOLD:
                st.success(f"Prediction: **{label.upper()}**")
            else:
                st.warning(f"Low confidence: top prediction is **{label.upper()}**. Try a clearer image.")
            st.metric("Confidence", f"{confidence * 100:.1f}%")
            st.subheader("Top 3 predictions")
            for name, score in top_predictions:
                st.write(f"**{name.upper()}** — {score * 100:.1f}%")
                st.progress(min(max(score, 0.0), 1.0))

else:
    st.write("Allow camera access when your browser asks. Hold a single alphabet sign steady; predictions update as frames arrive.")
    try:
        import av
        import cv2
        from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

        class ASLVideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.last_label = ""
                self.last_confidence = 0.0
                self.lock = threading.Lock()
                self.frame_count = 0

            def recv(self, frame):
                bgr = frame.to_ndarray(format="bgr24")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                self.frame_count += 1
                # Predict every third frame to reduce CPU load on free hosting.
                if self.frame_count % 3 == 0:
                    try:
                        pil_image = Image.fromarray(rgb)
                        label, confidence, _ = predict_image(pil_image, model, class_names)
                        with self.lock:
                            self.last_label = label.upper()
                            self.last_confidence = confidence
                    except Exception:
                        pass
                with self.lock:
                    label = self.last_label
                    confidence = self.last_confidence
                text = f"{label}  {confidence * 100:.1f}%" if label else "Show an ASL alphabet sign"
                cv2.rectangle(bgr, (8, 8), (min(bgr.shape[1] - 8, 340), 54), (0, 0, 0), -1)
                cv2.putText(bgr, text, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                return av.VideoFrame.from_ndarray(bgr, format="bgr24")

        webrtc_streamer(
            key="asl-realtime-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=ASLVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.caption("For best results, use HTTPS (the deployed Streamlit link), allow camera permission, and keep the hand visible. This recognizes static letters, not moving words or sentences.")
    except Exception as exc:
        st.error("Real-time camera mode could not start. Check the deployment dependencies and refresh the app.")
        st.exception(exc)

with st.expander("About this project"):
    st.write("This project recognizes static ASL alphabet signs. It does not interpret dynamic signs, words, or complete sentences. Results vary with lighting, camera angle, hand position, and differences from the training dataset.")
    st.caption("Model: MobileNetV2 · Input size: 160 × 160 · Classes: 29")
